"""Lokalny most BiRefNet -> ComfyUI. Tylko Python standard library.

Użycie:
    python comfy_birefnet_bridge.py "C:\\zdjecia\\test.jpg"
    python comfy_birefnet_bridge.py "C:\\zdjecia\\test.png" --server http://127.0.0.1:8188

Wymaga pliku 01_BiRefNet_General_PROMPT_API.json w tym samym folderze.
"""
import argparse
import json
import mimetypes
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

WORKFLOW = Path(__file__).with_name('01_BiRefNet_General_PROMPT_API.json')


def request_json(url, payload=None, timeout=30):
    body = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=body, headers=({'Content-Type':'application/json'} if body else {}))
    with urllib.request.urlopen(req, timeout=timeout) as reply:
        return json.loads(reply.read().decode('utf-8'))


def upload_image(server, path):
    boundary = '----ComfyBridge' + uuid.uuid4().hex
    name = uuid.uuid4().hex + path.suffix.lower()
    mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    header = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
              f'Content-Type: {mime}\r\n\r\n').encode('utf-8')
    # Dla dużych obrazów unikamy ręcznego kopiowania plików do folderu ComfyUI;
    # upload odbywa się lokalnie przez API.
    data = (header + path.read_bytes() + b'\r\n' +
            f'--{boundary}\r\nContent-Disposition: form-data; name="type"\r\n\r\ninput\r\n'.encode('utf-8') +
            f'--{boundary}--\r\n'.encode('utf-8'))
    req = urllib.request.Request(server+'/upload/image',data=data,headers={'Content-Type':f'multipart/form-data; boundary={boundary}'},method='POST')
    with urllib.request.urlopen(req,timeout=120) as reply:
        result=json.loads(reply.read().decode('utf-8'))
    if not result.get('name'):
        raise RuntimeError(f'ComfyUI nie zwrócił nazwy pliku: {result}')
    return result


def main():
    p=argparse.ArgumentParser(description='Wycinanie tła BiRefNet przez lokalne API ComfyUI')
    p.add_argument('image',type=Path,help='Obraz JPG/PNG/WEBP do wycięcia')
    p.add_argument('--server',default='http://127.0.0.1:8188',help='Adres ComfyUI (domyślnie 8188)')
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('wyniki_birefnet'),help='Folder wyników')
    p.add_argument('--timeout',type=int,default=600,help='Limit oczekiwania w sekundach')
    args=p.parse_args()
    image=args.image.expanduser().resolve()
    if not image.is_file():
        p.error(f'Nie ma pliku: {image}')
    if image.suffix.lower() not in ('.png','.jpg','.jpeg','.webp'):
        p.error('Obsługiwane wejście: PNG, JPG, WEBP')
    if not WORKFLOW.is_file():
        p.error(f'Brak workflow: {WORKFLOW}')
    server=args.server.rstrip('/')
    try:
        info=request_json(server+'/object_info/RembgByBiRefNet')
        if 'RembgByBiRefNet' not in info:
            raise RuntimeError('Nie znaleziono RembgByBiRefNet na serwerze')
        print('1/4 Połączenie ComfyUI OK. Wysyłam zdjęcie...')
        uploaded=upload_image(server,image)
        workflow=json.loads(WORKFLOW.read_text(encoding='utf-8'))
        if '2' not in workflow or workflow['2'].get('class_type')!='LoadImage':
            raise RuntimeError('Nieprawidłowy plik workflow')
        subfolder=uploaded.get('subfolder','')
        workflow['2']['inputs']['image']=(subfolder.rstrip('/')+'/' if subfolder else '')+uploaded['name']
        # Użyj pojedynczego ID, aby odróżnić wyniki współbieżnych zadań.
        run_id=uuid.uuid4().hex[:10]
        workflow['9']['inputs']['filename_prefix']=f'BiRefNet/cutout_{run_id}'
        workflow['11']['inputs']['filename_prefix']=f'BiRefNet/mask_{run_id}'
        print('2/4 Uruchamiam BiRefNet General...')
        queued=request_json(server+'/prompt',{'prompt':workflow,'client_id':'photocraft_bridge_'+run_id},timeout=60)
        prompt_id=queued.get('prompt_id')
        if not prompt_id:
            raise RuntimeError('ComfyUI odrzucił workflow: '+json.dumps(queued,ensure_ascii=False)[:1500])
        deadline=time.monotonic()+args.timeout
        history=None
        while time.monotonic()<deadline:
            response=request_json(server+'/history/'+urllib.parse.quote(prompt_id),timeout=30)
            history=response.get(prompt_id)
            if history:
                break
            time.sleep(2)
        if not history:
            raise TimeoutError(f'Upłynął limit oczekiwania; zadanie {prompt_id} może nadal działać w ComfyUI')
        if (history.get('status') or {}).get('status_str')=='error':
            raise RuntimeError('ComfyUI zgłosił błąd: '+json.dumps(history.get('status'),ensure_ascii=False)[:1500])
        print('3/4 Odbieram PNG i maskę...')
        outputs=history.get('outputs') or {}
        target=args.output.expanduser().resolve()/('birefnet_'+run_id)
        target.mkdir(parents=True,exist_ok=True)
        copied=[]
        for node_id,stem in [('9','cutout'),('11','mask')]:
            images=(outputs.get(node_id) or {}).get('images') or []
            if not images:
                raise RuntimeError(f'Brak obrazu wyjściowego w węźle {node_id}. Sprawdź workflow i historię ComfyUI.')
            for i,item in enumerate(images):
                params=urllib.parse.urlencode({'filename':item['filename'],'subfolder':item.get('subfolder',''),'type':item.get('type','output')})
                req=urllib.request.Request(server+'/view?'+params)
                with urllib.request.urlopen(req,timeout=120) as reply:
                    ext=Path(item['filename']).suffix.lower() or '.png'
                    output=target/(stem+(f'_{i+1}' if len(images)>1 else '')+ext)
                    with output.open('wb') as f:
                        shutil.copyfileobj(reply,f)
                    copied.append(output)
        print('4/4 Gotowe! Pliki:')
        for path in copied:
            print('   '+str(path))
        print('Uwaga: sprawdź przezroczystość cutout PNG; maska jest zapisana osobno.')
        return 0
    except (urllib.error.URLError,TimeoutError,ValueError,RuntimeError,OSError,KeyError) as e:
        print('BŁĄD:',e,file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
