# -*- coding: utf-8 -*-
"""PhotoCraft AI: test kompozycji i formatu do druku, WYŁĄCZNIE ODCZYT.
Czyta aktywny dokument MCP, obraz z dysku; Qwen3-VL wskazuje bbox grupy,
program wylicza 3 proporcje i zapisuje podglądy. Nigdy nie uruchamia command_run.
"""
import asyncio
import base64
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
MCP_URL = 'http://127.0.0.1:8766/mcp'
OLLAMA_URL = 'http://127.0.0.1:11434/api/chat'
MODEL = 'qwen3-vl:8b-instruct'
REPORT = ROOT / 'raport_propozycji_kadru.txt'
OUTPUT = ROOT / 'podglad_trzech_kadrow.png'
OUTPUT_JSON = ROOT / 'propozycje_kadrow.json'


def normalize(value):
    value = unicodedata.normalize('NFKD', str(value).casefold())
    return re.sub(r'[^a-z0-9]+', '', ''.join(x for x in value if not unicodedata.combining(x)))


def error_details(exc):
    if isinstance(exc, BaseExceptionGroup):
        return ' | '.join(error_details(e) for e in exc.exceptions)
    return f'{type(exc).__name__}: {exc}'


def decode_mcp(r):
    if getattr(r, 'isError', False):
        raise RuntimeError('Błąd MCP: ' + ' '.join(getattr(x, 'text', '') for x in r.content)[:400])
    sc = getattr(r, 'structuredContent', None)
    if isinstance(sc, dict) and {'name', 'width', 'height'} <= sc.keys():
        return sc
    for item in r.content:
        if getattr(item, 'type', None) == 'text':
            try:
                obj = json.loads(item.text)
                if isinstance(obj, dict):
                    return obj
            except (ValueError, TypeError):
                pass
    raise ValueError('Nie można odczytać danych dokumentu')


async def inspect_document():
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    async with streamable_http_client(MCP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            return decode_mcp(await session.call_tool('doc_inspect', arguments={}))


def vision_subject_bbox(img):
    """Bounding box osób 0..1000 (nie gotowy crop). Wymaga lokalnego Ollama."""
    image = ImageOps.exif_transpose(img).convert('RGB')
    image.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, 'JPEG', quality=83)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    task = (
        '/no_think\nObejrzyj zdjęcie i znajdź wszystkie osoby będące głównym tematem fotografii. '
        'Zwróć JEDYNIE JSON: {"box":[left,top,right,bottom],"note":"krótka uwaga"}. '
        'Współrzędne w skali 0..1000 względem CAŁEGO obrazu (lewa krawędź=0, górna=0). '
        'Box ma obejmować wszystkie osoby od głowy po stopy oraz wystawione dłonie, telefony, '
        'o ile są widoczne. Nie obejmuj niepotrzebnego nieba czy innych elementów tła. '
        'Nie wymyślaj osób, nie edytuj obrazu. Bez markdown i bez dodatkowych wyjaśnień.'
    )
    payload = {'model': MODEL, 'stream': False, 'think': False,
               'options': {'num_ctx': 4096, 'num_predict': 220, 'temperature': 0},
               'format': 'json',
               'messages': [{'role': 'user', 'content': task, 'images': [encoded]}]}
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=130) as resp:
        reply = json.loads(resp.read().decode('utf-8'))
    txt = reply.get('message', {}).get('content', '').strip()
    obj = json.loads(txt)
    box = obj.get('box')
    if not isinstance(box, list) or len(box) != 4 or any(type(n) not in (int, float) for n in box):
        raise ValueError(f'Qwen nie podał wiarygodnego bbox: {txt[:250]}')
    l,t,r,b = [float(n) for n in box]
    if not (0 <= l < r <= 1000 and 0 <= t < b <= 1000 and (r-l)>=80 and (b-t)>=90):
        raise ValueError('Nieprawidłowe współrzędne Qwen: ' + str(box))
    return (l/1000*img.width, t/1000*img.height, r/1000*img.width, b/1000*img.height), str(obj.get('note', ''))[:160]


def font(size):
    for p in (r'C:\Windows\Fonts\arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try: return ImageFont.truetype(p, size)
        except OSError: pass
    return ImageFont.load_default()


def choose_image():
    preferred = [ROOT/'przyjacielski_selfie_na_parkowej_ławce.png',
                 ROOT/'Przyjacielski selfie na parkowej ławce.png']
    for f in preferred:
        if f.exists(): return f
    # Dopuszczamy dowolną kopię tego zdjęcia, w tym plik bez polskich znaków.
    hits = [p for p in ROOT.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')
            and 'przyjacielski' in normalize(p.stem) and 'parkowej' in normalize(p.stem)]
    if len(hits) == 1: return hits[0]
    raise FileNotFoundError('Skopiuj oryginalne zdjęcie testowe PNG do folderu obok skryptu. '
                            'Skrypt nie obsługuje przyciętego pliku z PhotoCraft.')



# Rozpoznanie intencji nie wymaga kolejnego zapytania do modelu:
# celowo przewidywalny parser, bez ryzyka, że AI zignoruje jawny format.
PRINT_FORMATS = [(13,18),(15,21),(18,24),(20,30),(21,30),(30,40),(30,45),
                 (40,50),(40,60),(50,70),(60,80),(70,100)]


def parse_request(prompt):
    clean = unicodedata.normalize('NFKD', prompt.lower())
    clean = ''.join(c for c in clean if not unicodedata.combining(c))
    # Explicit physical dimensions, e.g. 50x70 cm, 30 × 40 cm.
    match = re.search(r'(\d{1,3}(?:[.,]\d+)?)\s*[x×]\s*(\d{1,3}(?:[.,]\d+)?)\s*(cm|mm)\b', clean)
    print_cm = None
    if match:
        a,b = float(match.group(1).replace(',','.')),float(match.group(2).replace(',','.'))
        scale = 0.1 if match.group(3)=='mm' else 1.0
        print_cm = (a*scale,b*scale)
    else:
        # In print context accept familiar nominal sizes even without "cm".
        for a,b in PRINT_FORMATS:
            if re.search(r'(?<!\d)'+str(a)+r'\s*[x×]\s*'+str(b)+r'(?!\d)',clean):
                print_cm=(float(a),float(b)); break
    strong = ('mocn' in clean or 'trojpodzial' in clean or 'trzec' in clean)
    anchor = 'center'
    if strong:
        right = 'praw' in clean
        left = 'lew' in clean
        top = 'gorn' in clean or 'gor' in clean
        bottom = 'doln' in clean or 'dol' in clean
        # When only right/left given, choose top-third by convention;
        # when only bottom/top, choose right-third by convention.
        anchor=('right' if right else 'left' if left else 'right')+'_'+('bottom' if bottom else 'top')
    elif 'po lewej' in clean or 'z lewej' in clean:
        anchor='left_center'
    elif 'po prawej' in clean or 'z prawej' in clean:
        anchor='right_center'
    if any(k in clean for k in ('ciasn','blisk','zbliz','portret')):
        style='tight'
    elif any(k in clean for k in ('szerok','luzn','otoczen','przestrzen','krajobraz')):
        style='wide'
    else:
        style='natural'
    orient = 'portrait' if any(k in clean for k in ('pionow','pionie','pionowy')) else 'landscape' if any(k in clean for k in ('poziom','poziomie','poziomy')) else None
    ratio = None
    ratio_match = re.search(r'(?<!\d)([1-9])\s*[:/]\s*([1-9])(?!\d)',clean)
    if ratio_match:
        ratio = int(ratio_match.group(1))/int(ratio_match.group(2))
    if print_cm:
        a,b=print_cm
        ratio=max(a,b)/min(a,b)
        if orient=='portrait' or (orient is None and any(k in clean for k in ('pionow','pionie'))): ratio=1/ratio
        elif orient is None and not any(k in clean for k in ('poziom','poziomie')):
            # Typical print convention "30x40" given without orientation: ask the user.
            # For this preview choose landscape but state the assumption.
            ratio=ratio
    elif ratio and orient=='portrait' and ratio>1:
        ratio=1/ratio
    elif ratio and orient=='landscape' and ratio<1:
        ratio=1/ratio
    elif ratio is None and orient=='portrait': ratio=4/5
    return {'prompt':prompt,'print_cm':print_cm,'ratio':ratio,'style':style,
            'anchor':anchor,'orientation':orient,'explicit_ratio':bool(ratio_match)}


def anchor_point(anchor):
    if anchor=='center': return (0.5,0.5)
    x=1/3 if 'left' in anchor else 2/3 if 'right' in anchor else .5
    y=1/3 if 'top' in anchor else 2/3 if 'bottom' in anchor else .5
    return (x,y)


def plan_frame(bbox,W,H,ratio,coverage,anchor='center'):
    """Large-to-small constrained framing; never cuts detected subject.
    Includes a small measured buffer; if requested anchor is infeasible,
    place subject as close as possible and report achieved normalized point.
    """
    if not (.3 <= ratio <= 3.5): return None
    l,t,r,b=map(float,bbox)
    margin_x = max(10, .022*W)
    margin_y = max(10, .022*H)
    sl,sr = max(0,l-margin_x),min(W,r+margin_x)
    st,sb = max(0,t-margin_y),min(H,b+margin_y)
    maxw=min(W,H*ratio)
    maxh=maxw/ratio
    minimum=max((sr-sl)/maxw,(sb-st)/maxh)
    if minimum>1.00001: return None
    cov=min(1,max(coverage,minimum))
    w,h=maxw*cov,maxh*cov
    # subject centroid relative to crop at desired fraction; clamp to allowed
    # feasible positions so entire margin-padded bbox remains in the crop.
    ax,ay=anchor_point(anchor)
    cx,cy=(l+r)*.5,(t+b)*.5
    target_x=cx-ax*w
    target_y=cy-ay*h
    minx=max(0,sr-w); maxx=min(sl,W-w)
    miny=max(0,sb-h); maxy=min(st,H-h)
    if minx>maxx+.001 or miny>maxy+.001: return None
    x=max(minx,min(maxx,target_x));y=max(miny,min(maxy,target_y))
    # Integer rectangle rounding, slightly grow rather than lose subject margin.
    iw=min(W,int(round(w)));ih=min(H,int(round(h)))
    ix=max(0,min(W-iw,int(round(x))));iy=max(0,min(H-ih,int(round(y))))
    if ix>sl+1 or iy>st+1 or ix+iw<sr-1 or iy+ih<sb-1: return None
    achieved=((cx-ix)/iw,(cy-iy)/ih)
    deviation=((achieved[0]-ax)**2+(achieved[1]-ay)**2)**0.5
    return {'rect':(ix,iy,ix+iw,iy+ih), 'requested_anchor':(ax,ay),
            'achieved_anchor':tuple(round(z,3) for z in achieved),
            'anchor_deviation':round(deviation,3),'coverage':round(cov,3),
            'coverage_limited':cov>coverage+.001}


def print_assessment(rect, print_cm):
    if not print_cm or not rect: return None
    x0,y0,x1,y1=rect
    w,h=x1-x0,y1-y0
    a,b=print_cm
    # Choose orientation of page to match crop, not user convention.
    cmw,cmh=(max(a,b),min(a,b)) if w>=h else (min(a,b),max(a,b))
    xppi=w/(cmw/2.54); yppi=h/(cmh/2.54)
    return {'rozmiar_cm':[cmw,cmh],'ppi_x':round(xppi,1),'ppi_y':round(yppi,1),
            'ppi_ograniczajace':round(min(xppi,yppi),1),
            'uwaga':'PPI z oryginalnych pikseli; bez resamplingu i bez oceny jakości wydruku'}


def render_previews(img, proposals, output):
    W,H=img.size; tilew=530; tileh=round(H*tilew/W); header=94
    canvas=Image.new('RGB',(tilew*len(proposals),tileh+header),'#f4f5f7')
    draw=ImageDraw.Draw(canvas)
    for i,p in enumerate(proposals):
        ox=i*tilew
        draw.text((ox+13,9),p['nazwa'],fill='#172533',font=font(22))
        if not p['mozliwe']:
            draw.text((ox+13,45),'Brak bezpiecznego kadru',fill='#a11b1b',font=font(17))
            continue
        x0,y0,x1,y1=p['rect']; iw=x1-x0; ih=y1-y0
        draw.text((ox+13,40),f'x={x0} y={y0}   {iw} x {ih} px',fill='#374b63',font=font(15))
        if p.get('druk'):
            draw.text((ox+13,65),f"Wydruk: {p['druk']['ppi_ograniczajace']:.0f} PPI",fill='#374b63',font=font(15))
        elif p['anchor_deviation']>0.1:
            draw.text((ox+13,65),'Mocny punkt: tylko przybliżenie',fill='#aa4e08',font=font(14))
        base=img.convert('RGBA')
        dark=Image.new('RGBA',(W,H),(0,0,0,125)); d=ImageDraw.Draw(dark)
        d.rectangle((x0,y0,x1-1,y1-1),fill=(0,0,0,0))
        combined=Image.alpha_composite(base,dark).convert('RGB')
        bd=ImageDraw.Draw(combined)
        bd.rectangle((x0,y0,x1-1,y1-1),outline='#ff7b00',width=5)
        # Rule of thirds inside the orange crop area.
        for frac in (1/3,2/3):
            xx=round(x0+frac*iw);yy=round(y0+frac*ih)
            bd.line((xx,y0,xx,y1),fill='#fff3a0',width=2)
            bd.line((x0,yy,x1,yy),fill='#fff3a0',width=2)
        canvas.paste(combined.resize((tilew,tileh),Image.Resampling.LANCZOS),(ox,header))
    canvas.save(output)


def run_proposal(request, img, doc, bbox, note, iteration):
    """Jeden podgląd z analizy w pamięci; nie komunikuje się z MCP ani Qwen."""
    intent = parse_request(request)
    lines = ['PHOTOCRAFT AI: KADROWANIE WG POLECENIA v3.1 (TYLKO PODGLĄD)',
             'Data: '+datetime.now().isoformat(timespec='seconds'),
             'Sesja: '+str(iteration),
             'Komendy edycyjne MCP: ZERO. Brak zapisu i zmian w dokumencie.',
             'Polecenie: '+(request or '[domyślne trzy proporcje]'),
             'Interpretacja: '+json.dumps(intent, ensure_ascii=False),
             f"Źródło: {doc.get('name')}, {img.width} x {img.height}",
             'Qwen3-VL bbox osób (analiza jednorazowa): '+str(tuple(round(v) for v in bbox)),
             'Opis Qwen: '+note]
    if intent['ratio']:
        values={'wide':[('Bardzo szeroki',1.0),('Szeroki',.97),('Naturalny',.91)],
                'natural':[('Szeroki',.98),('Naturalny',.90),('Ciaśniejszy',.81)],
                'tight':[('Naturalny',.91),('Ciaśniejszy',.82),('Najbliższy bezpieczny',.72)]}[intent['style']]
        choices=[(name,intent['ratio'],cov) for name,cov in values]
    else:
        cov={'wide':.99,'natural':.92,'tight':.81}[intent['style']]
        choices=[('Poziomy 3:2',1.5,cov),('Kwadrat 1:1',1.0,cov),('Pionowy 4:5',.8,cov)]
    proposals=[]
    seen=set()
    for label,ratio,coverage in choices:
        frame=plan_frame(bbox,img.width,img.height,ratio,coverage,intent['anchor'])
        if frame is None:
            lines.append(f'{label}: brak bezpiecznego kadru'); continue
        if frame['rect'] in seen:
            lines.append(f'{label}: pominięto — identyczny kadr jak wcześniejsza propozycja'); continue
        seen.add(frame['rect'])
        x0,y0,x1,y1=frame['rect']
        param={'x':x0,'y':y0,'width':x1-x0,'height':y1-y0,'angle':0,
               'deleteCroppedPixels':False,'contentAware':False}
        pp=print_assessment(frame['rect'],intent['print_cm'])
        proposal={'nazwa':label,'mozliwe':True,'rect':list(frame['rect']),
                  'parametry_image_crop':param,'druk':pp,**{k:v for k,v in frame.items() if k!='rect'}}
        proposals.append(proposal)
        lines.append(label+': '+json.dumps(param,ensure_ascii=False))
        if intent['anchor']!='center':
            msg=f"  Cel mocnego punktu {frame['requested_anchor']}, uzyskano {frame['achieved_anchor']}, odchylenie {frame['anchor_deviation']:.3f}."
            if frame['anchor_deviation']>.1: msg+=' UWAGA: mocny punkt jest tylko przybliżony.'
            lines.append(msg)
        if frame['coverage_limited']: lines.append('  UWAGA: zwiększono kadr, żeby nie uciąć grupy.')
        if pp:
            lines.append('  Druk: '+json.dumps(pp,ensure_ascii=False))
            if pp['ppi_ograniczajace']<100:
                lines.append('  UWAGA: bardzo niska rozdzielczość na wskazany wydruk.')
    if not proposals:
        lines.append('Brak bezpiecznej propozycji. Nie utworzono podglądu.')
    else:
        render_previews(img,proposals,OUTPUT)
        OUTPUT_JSON.write_text(json.dumps({'document':doc.get('name'),'size':[img.width,img.height],
            'mode':'ONLY_PREVIEW','intent':intent,'source':'qwen3-vl bbox + deterministic framing',
            'subject_bbox_px':[round(v,2) for v in bbox], 'proposals':proposals},
            ensure_ascii=False,indent=2),encoding='utf-8')
        lines.append('Podgląd: '+str(OUTPUT))
        lines.append('JSON: '+str(OUTPUT_JSON))
    lines += ['UWAGA: PPI nie jest gwarancją jakości wydruku.',
              'UWAGA: bbox Qwen może być niedokładny; zawsze sprawdź podgląd.',
              'UWAGA: mocny punkt dotyczy środka wykrytej grupy, nie pojedynczej twarzy.']
    report='\n'.join(lines)+'\n'
    REPORT.write_text(report,encoding='utf-8-sig')
    # Jednocześnie zachowaj kolejne wyniki, aby niczego nie nadpisać.
    history=ROOT/'historia_kadrow'
    history.mkdir(exist_ok=True)
    prefix=f'kadr_{iteration:03d}'
    (history/(prefix+'.txt')).write_text(report,encoding='utf-8-sig')
    if proposals:
        import shutil
        shutil.copyfile(OUTPUT,history/(prefix+'.png'))
        shutil.copyfile(OUTPUT_JSON,history/(prefix+'.json'))
    print('\n'.join(lines))
    print('\nZapisano w historii:',history/prefix)
    return len(proposals)


def main():
    print('PhotoCraft AI — kadrowanie v3.1, wiele poleceń w jednej sesji (TYLKO PODGLĄD)')
    print('Wpisz polecenie, Enter = trzy proporcje, KONIEC = wyjście.')
    print('Analiza Qwen wykonywana jest tylko raz podczas startu.')
    try:
        path=choose_image()
        with Image.open(path) as src:
            img=ImageOps.exif_transpose(src).convert('RGB')
        doc=asyncio.run(inspect_document())
        if doc.get('width')!=img.width or doc.get('height')!=img.height:
            raise ValueError('Aktywny dokument ma inne wymiary niż oryginalny plik; otwórz oryginał.')
        if normalize(Path(str(doc.get('name',''))).stem)!=normalize(path.stem):
            raise ValueError('Aktywny dokument różni się od pliku testowego.')
        print('Rozpoznaję osoby na zdjęciu przez Qwen3-VL...')
        bbox,note=vision_subject_bbox(img)
        print('Analiza zakończona. Zaznaczony obszar:',tuple(round(x) for x in bbox))
    except Exception as exc:
        print('PRZERWANO:',error_details(exc))
        return
    iteration=0
    while True:
        try:
            request=input('\nNowy kadr (KONIEC = zamknij): ').strip()
        except (KeyboardInterrupt,EOFError):
            print('\nZakończono sesję.')
            break
        if normalize(request) in ('koniec','exit','quit','q','wyjdz'):
            print('Zakończono sesję.')
            break
        iteration+=1
        if OUTPUT.exists(): OUTPUT.unlink()
        try:
            count=run_proposal(request,img,doc,bbox,note,iteration)
            if count:
                print('Podgląd zapisany. Otwórz plik:',OUTPUT)
                print('Możesz wpisać następne polecenie bez uruchamiania skryptu od nowa.')
        except Exception as exc:
            print('BŁĄD propozycji:',error_details(exc))

if __name__=='__main__': main()
