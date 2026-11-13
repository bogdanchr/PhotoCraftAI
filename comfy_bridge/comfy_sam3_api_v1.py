"""Lokalny SAM3/ComfyUI: pełny obraz + maska + PNG cutout, bez PhotoCraft.

Korzysta z API JSON użytkownika; nie przerabia zapisanej kopii workflow.
Działa tylko lokalnie, z biblioteką standardową. Model SAM3 działa w ComfyUI.
"""
import argparse
import copy
import json
import mimetypes
import shutil
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

API_URL = 'http://127.0.0.1:8188'
WORKFLOW = Path(__file__).with_name('SAM3_Wycinanie_z_opisu_i_laczenie_masek.json')
EXPECTED = {'1':'LoadImage', '126':'MaskComposite', '133':'MaskBlur+', '129':'JoinImageWithAlpha',
            '114:78':'CLIPTextEncode', '119:118':'CLIPTextEncode'}


def json_request(url, data=None, timeout=30):
    payload = None if data is None else json.dumps(data).encode('utf-8')
    request = urllib.request.Request(url, data=payload,
                                     headers={'Content-Type':'application/json'} if payload else {})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.load(reply)


def upload_image(base, path):
    boundary = '----PhotoCraftSAM3' + uuid.uuid4().hex
    name = 'sam3_' + uuid.uuid4().hex[:16] + path.suffix.lower()
    mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    fields = [
        (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
         f'Content-Type: {mime}\r\n\r\n').encode() + path.read_bytes() + b'\r\n',
        (f'--{boundary}\r\nContent-Disposition: form-data; name="type"\r\n\r\ninput\r\n').encode(),
        (f'--{boundary}--\r\n').encode(),
    ]
    req = urllib.request.Request(base+'/upload/image', data=b''.join(fields), method='POST',
                                 headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(req, timeout=120) as reply:
        result = json.load(reply)
    if not isinstance(result, dict) or not result.get('name'):
        raise RuntimeError(f'Nieudany upload do ComfyUI: {result}')
    return (result.get('subfolder','').strip('/') + '/' if result.get('subfolder') else '') + result['name']


def prompt_parts(description):
    # Tylko najprostsze, jawne tłumaczenia; nie zgadujemy semantyki osób w grupie.
    phrases = {
        'blondynka':'blonde woman', 'blondynkę':'blonde woman',
        'telefon':'phone', 'telefonem':'phone', 'telefonu':'phone',
        'kobieta':'woman', 'kobietę':'woman',
        'mężczyzna':'man', 'mężczyznę':'man',
        'pies':'dog', 'psa':'dog', 'kot':'cat', 'kota':'cat',
    }
    raw = description.strip()
    if not raw or len(raw)>400:
        raise ValueError('Opis musi mieć 1–400 znaków.')
    # Dla „blondynka z telefonem” rozdziel opisy — SAM3 nie zawsze rozumie relacje.
    import re
    raw = re.sub(r'\b(blondynka|blondynkę)\s+z\s+telefonem\b', r'\1 + telefon', raw, flags=re.I)
    # Jawny separator '+' — tylko dwa przebiegi w tym workflow.
    parts = [part.strip().strip('.! ') for part in raw.split('+')]
    if not (1<=len(parts)<=2) or any(not p or len(p)>200 for p in parts):
        raise ValueError('Podaj 1 lub 2 opisy, oddzielając je znakiem + (np. blonde woman + phone).')
    converted = [phrases.get(p.casefold(),p) for p in parts]
    return converted


def build_prompt(workflow, uploaded, parts, token):
    graph = copy.deepcopy(workflow)
    for key, cls in EXPECTED.items():
        if key not in graph or graph[key].get('class_type') != cls:
            raise ValueError(f'Workflow nie pasuje do oczekiwanego schematu SAM3: węzeł {key} ({cls}).')
    graph['1']['inputs']['image']=uploaded
    graph['119:118']['inputs']['text']=parts[0]
    graph['114:78']['inputs']['text']=parts[1] if len(parts)>1 else parts[0]
    # 131 zapisuje przezroczysty cutout. Dodatkowa gałąź zapisuje prawdziwą maskę
    # z węzła po Grow+Blur, przed odwróceniem do alpha (białe = widoczne).
    graph['131']['inputs']['filename_prefix']='PhotoCraft_SAM3/cutout_'+token
    graph['sam3_mask_image'] = {'class_type':'MaskToImage', 'inputs':{'mask':['133',0]}}
    graph['sam3_mask_save'] = {'class_type':'SaveImage',
                               'inputs':{'filename_prefix':'PhotoCraft_SAM3/mask_'+token,
                                         'images':['sam3_mask_image',0]}}
    return graph


def save_output(base, output_info, destination):
    images = (output_info or {}).get('images') or []
    if len(images)!=1:
        raise RuntimeError(f'ComfyUI zwrócił {len(images)} obrazów, oczekiwano jednego: {destination.name}')
    item=images[0]
    if not item.get('filename'):
        raise RuntimeError('ComfyUI zwrócił brak nazwy wyniku.')
    query=urllib.parse.urlencode({k:item.get(k, '') for k in ('filename','subfolder','type')})
    with urllib.request.urlopen(base+'/view?'+query,timeout=120) as response:
        with destination.open('wb') as f:
            shutil.copyfileobj(response,f)
    if destination.stat().st_size<100:
        raise RuntimeError('Pobrano pusty/niepełny plik '+str(destination))
    with destination.open('rb') as f:
        if f.read(8) != b'\x89PNG\r\n\x1a\n':
            raise RuntimeError('Wynik nie jest prawidłowym PNG: '+str(destination))


def run(image, description, output_dir, base=API_URL, timeout=900):
    image=Path(image)
    if not image.is_file() or image.suffix.casefold() not in ('.png','.jpg','.jpeg','.webp'):
        raise FileNotFoundError('Nie znaleziono obrazu PNG/JPG/WebP: '+str(image))
    if not WORKFLOW.is_file():
        raise FileNotFoundError('Brak workflow: '+str(WORKFLOW))
    parts=prompt_parts(description)
    graph_template=json.loads(WORKFLOW.read_text(encoding='utf-8-sig'))
    token=uuid.uuid4().hex[:12]
    base=base.rstrip('/')
    # Sprawdź, czy lokalny ComfyUI ma kluczowe węzły; błędy promptu są potem czytelne.
    info=json_request(base+'/object_info/SAM3_Detect',timeout=20)
    if 'SAM3_Detect' not in info:
        raise RuntimeError('ComfyUI nie udostępnia SAM3_Detect.')
    uploaded=upload_image(base,image)
    graph=build_prompt(graph_template,uploaded,parts,token)
    queued=json_request(base+'/prompt',{'prompt':graph,'client_id':'pc_sam3_'+token},timeout=90)
    if queued.get('node_errors'):
        raise RuntimeError('Błędy walidacji workflow: '+json.dumps(queued['node_errors'],ensure_ascii=False)[:2500])
    prompt_id=queued.get('prompt_id')
    if not prompt_id:
        raise RuntimeError('ComfyUI nie uruchomił SAM3: '+json.dumps(queued,ensure_ascii=False)[:2500])
    print('SAM3: opis(y): '+repr(parts),flush=True)
    print('SAM3: zadanie '+prompt_id,flush=True)
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        history=json_request(base+'/history/'+urllib.parse.quote(prompt_id),timeout=45).get(prompt_id)
        if history:
            status=history.get('status') or {}
            if status.get('status_str')=='error' or not (history.get('outputs') or {}).get('sam3_mask_save'):
                raise RuntimeError('ComfyUI nie ukończył SAM3: '+json.dumps(status,ensure_ascii=False)[:2500])
            break
        time.sleep(2)
    else:
        raise TimeoutError('Czas oczekiwania minął; nie uruchamiaj ponownie bez sprawdzenia kolejki ComfyUI.')
    output=Path(output_dir).resolve()/('sam3_'+token)
    output.mkdir(parents=True,exist_ok=False)
    cutout=output/'cutout.png'; mask=output/'mask.png'
    save_output(base,history['outputs'].get('131'),cutout)
    save_output(base,history['outputs'].get('sam3_mask_save'),mask)
    print('Wyciecie: '+str(cutout),flush=True)
    print('Maska: '+str(mask),flush=True)
    return cutout,mask


def main():
    parser=argparse.ArgumentParser(description='SAM3: zapis cutout PNG i maski.')
    parser.add_argument('--image',required=True)
    parser.add_argument('--prompt',required=True)
    parser.add_argument('--output',default=str(Path(__file__).with_name('wyniki_sam3')))
    parser.add_argument('--server',default=API_URL)
    parser.add_argument('--timeout',type=int,default=900)
    parser.add_argument('--preview',action='store_true',help='Sprawdź opisy i graf bez połączenia z ComfyUI.')
    args=parser.parse_args()
    try:
        if args.preview:
            parts=prompt_parts(args.prompt)
            graph=build_prompt(json.loads(WORKFLOW.read_text(encoding='utf-8-sig')),'MOCK.png',parts,'test')
            print('Opis(y):',parts)
            print('Workflow:',len(graph),'węzłów; wejście i dwie gałęzie zapisu poprawne.')
            return 0
        run(args.image,args.prompt,args.output,args.server,args.timeout)
        return 0
    except Exception as exc:
        print('BŁĄD SAM3:',str(exc),file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
