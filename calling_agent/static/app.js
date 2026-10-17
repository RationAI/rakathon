import {inputConstraints, availableInputs} from './audio-routing.js';
import {GreetingPlayer} from './greeting.js';
const $ = (id) => document.getElementById(id);
let current = null, config = null, peer = null, media = null, channel = null;
let poll = null, closeTimer = null, connecting = false, active = false;
let captions = [], captionSpeaker = null;
const stateNames = {created:'Připraveno',connecting:'Připojování',ringing:'Vyzvání',active:'Probíhá',closing:'Ukončování',closed:'Ukončeno',failed:'Chyba spojení',unknown:'Ověřte stav spojení',interrupted:'Přerušeno'};
const reasonNames = {not_asked:'Nezjištěno',time:'Čas / práce',transport:'Doprava',fear:'Obavy',forgot:'Zapomenutí',no_invitation:'Pozvání nedorazilo',already_attended:'Již absolvovala',health:'Zdravotní důvody',other:'Jiný důvod',declined_to_answer:'Nechce uvést'};

const greeting = new GreetingPlayer({
  onStart: () => {
    $('audio').muted = true;
    captions.push({type:'session.output_transcript.delta', delta:config.opening});
    renderCaptions(captions);
  },
  onEnd: () => { $('audio').muted = false; }
});
function speechPreferences() {return {voice:$('voice-choice').value||config?.voice,tone:$('tone-choice').value||config?.tone};}
function lockSpeech(locked) {$('voice-choice').disabled=$('tone-choice').disabled=locked;}
async function prepareGreeting() {
  const preferences=speechPreferences();
  const key=JSON.stringify(preferences);
  return greeting.prepare(async () => {
    const response = await fetch(`/api/greeting?${new URLSearchParams(preferences)}`);
    if (!response.ok) throw new Error('Hlasový úvod se nepodařilo načíst. Zkuste hlasový test znovu.');
    return response.arrayBuffer();
  }, key);
}
function message(text='') { $('message').textContent = text; }
function node(tag, text, cls) { const el=document.createElement(tag); if(text!==undefined)el.textContent=text; if(cls)el.className=cls; return el; }
async function api(path, options={}) {
  const r=await fetch(`/api${path}`, {...options, headers:{'Content-Type':'application/json',...options.headers}});
  const data=await r.json(); if(!r.ok)throw new Error(typeof data.detail==='string'?data.detail:'Požadavek se nezdařil.'); return data;
}
function field(label, value) { const wrap=node('dl',undefined,'field'); wrap.append(node('dt',label),node('dd',value||'Zatím neuvedeno')); return wrap; }
function link(label, url) { const a=node('a',label); a.href=url; a.target='_blank'; a.rel='noopener noreferrer'; return a; }
function safeLink(label,url) { return /^https?:\/\//.test(url||'') ? link(label,url) : node('span',label); }
function render(row) {
  current=row; $('record-actions').hidden=false; $('call-status').textContent=stateNames[row.state]||row.state;
  const p=row.payload, target=$('outcome'); target.replaceChildren();
  if(row.mode==='demo') target.append(node('p','Syntetická ukázka · žádný skutečný hovor', 'status-note'));
  const consent={unknown:'Zatím nezjištěn',granted:'Souhlasí s rozhovorem a uložením odpovědí',declined:'Odmítla rozhovor',withdrawn:'Souhlas odvolán'};
  target.append(field('Souhlas',consent[row.consent]));
  if(row.opted_out)target.append(node('p','Nepřeje si další kontakt. Telefonní číslo je blokováno pro další volání v této aplikaci.','status-note'));
  if(p.interview){target.append(field('Obec bydliště',p.interview.municipality),field('Co uvedla jako překážku',p.interview.reason_quote || reasonNames[p.interview.reason_category]),field('Doprava',p.interview.travel_preferences));}
  if(p.selected_center){const c=p.selected_center; const card=node('div',undefined,'center-choice');card.append(node('p','VYBRANÉ PRACOVIŠTĚ','eyebrow'),node('strong',c.name),node('p',`${c.address}, ${c.postal_code} ${c.city}`),node('p',c.phone.join(' · ')),safeLink('Detail na mamo.cz ↗',c.source_url));target.append(card);}
  if(p.preferences)target.append(field('Preferované období',p.preferences.preferred_dates),field('Denní doba',p.preferences.preferred_times),field('Potřeby a dostupnost',p.preferences.accessibility_needs));
  if(p.handoff)target.append(node('p','Vyžaduje lidské dořešení. Živé přepojení není zapojeno.','status-note'));
  if(p.booking)target.append(field('Ukázkový termín',p.booking.slot.label),node('p','Ukázkové objednání potvrzeno. Skutečná rezervace nevznikla.','status-note'));
  if(p.request)target.append(node('p','Požadavek potvrzen. Čeká na ruční objednání; termín není rezervovaný.','status-note'));
  if(row.error)target.append(node('p',row.error,'status-note'));
  if(['unknown','interrupted'].includes(row.state)){const resolve=node('button','Ověřeno u poskytovatele: hovor skončil','secondary');resolve.addEventListener('click',async()=>{try{render(await api(`/sessions/${row.id}/resolve`,{method:'POST',body:JSON.stringify({remote_ended_confirmed:true})}));await refreshHistory();}catch(e){message(e.message);}});target.append(resolve);}
  $('delete').disabled=['connecting','ringing','active','closing','unknown','interrupted'].includes(row.state);
  $('stop').disabled=!(active||['connecting','ringing','active','closing','unknown','interrupted'].includes(row.state));
}
async function refreshHistory(){const rows=await api('/sessions');$('history').replaceChildren();for(const row of rows){const el=node('div',undefined,'history-row');const b=node('button',row.contact_label||`${row.mode==='phone'?'Telefonní':'Hlasový'} rozhovor · ${new Date(row.created_at).toLocaleString('cs-CZ')}`,'text-button');b.addEventListener('click',()=>{if(active||connecting)return message('Nejdříve ukončete aktivní hovor.');select(row.id);});el.append(b,node('span',row.payload.booking?'Ukázka potvrzena':row.payload.request?'Čeká na objednání':stateNames[row.state]||row.state));$('history').append(el);}if(!rows.length)$('history').append(node('p','Zatím žádné rozhovory.','empty'));}
async function select(id){clearInterval(poll);const row=await api(`/sessions/${id}`);render(row);renderCaptions(row.captions||[]);}
function renderCaptions(items){const target=$('captions');target.replaceChildren();let speaker='';for(const item of items){const who=item.type==='session.input_transcript.delta'?'Volající':'Klára';if(who!==speaker){target.append(node('span',who,'speaker'));speaker=who;}target.append(document.createTextNode(item.delta||''));}if(!items.length)target.append(node('p','Zde se objeví průběh hlasového rozhovoru.','empty'));target.scrollTop=target.scrollHeight;}
function onEvent(event){if(event.type==='session.input_transcript.delta' && event.delta?.trim())greeting.cancel();if(event.type==='session.started'){$('call-status').textContent='Probíhá';}if(event.type==='session.closed'){cleanup();refreshCurrent();refreshHistory();}if(['session.input_transcript.delta','session.output_transcript.delta'].includes(event.type)){captions.push(event);captions=captions.slice(-120);renderCaptions(captions);}if(event.type==='error')message(`Hlasová služba: ${event.error?.code||'chyba'}. Ověřte konfiguraci serveru.`);}
function cleanup(){lockSpeech(false);greeting.cancel();clearTimeout(closeTimer);clearInterval(poll);media?.getTracks().forEach(t=>t.stop());channel?.close();peer?.close();$('audio').srcObject=null;$('audio').hidden=true;media=channel=peer=null;active=connecting=false;document.body.classList.remove('live');$('start-voice').disabled=!config?.voice_ready;$('demo').disabled=false;$('dial').disabled=!config?.outbound_enabled;$('stop').disabled=true;}
async function refreshCurrent(){if(current){const row=await api(`/sessions/${current.id}`);render(row);if(!peer)renderCaptions(row.captions||[]);if(['closed','failed'].includes(row.state)&&active){cleanup();await refreshHistory();}}}
function startPoll(){clearInterval(poll);poll=setInterval(()=>refreshCurrent().catch(e=>message(e.message)),2000);}
function gatherIce(connection){if(connection.iceGatheringState==='complete')return Promise.resolve();return new Promise((resolve,reject)=>{const check=()=>{if(connection.iceGatheringState==='complete'){clearTimeout(timer);connection.removeEventListener('icegatheringstatechange',check);resolve();}};const timer=setTimeout(()=>{connection.removeEventListener('icegatheringstatechange',check);reject(new Error('Vypršel čas pro navázání zvukového spojení.'));},10000);connection.addEventListener('icegatheringstatechange',check);check();});}
function setBusy(){lockSpeech(true);connecting=true;$('start-voice').disabled=$('demo').disabled=$('dial').disabled=true;message();}
async function initialize() {
  try {
    config = await api('/config');
    for (const [id, options, fallback, saved] of [
      ['voice-choice', config.voices, config.voice, 'mamo-voice'],
      ['tone-choice', config.tones, config.tone, 'mamo-tone']
    ]) {
      const select=$(id);
      select.replaceChildren(...options.map(option => new Option(option.label, option.id)));
      const previous=localStorage.getItem(saved);
      select.value=options.some(option => option.id===previous)?previous:fallback;
      select.disabled=false;
      select.addEventListener('change',()=>localStorage.setItem(saved,select.value));
    }
    $('connection').textContent = 'Připraveno';
    $('center-count').textContent = config.directory_count;
    $('voice-model').textContent = config.voice_model;
    $('model-status').textContent = config.voice_ready ? 'Připraveno k hlasovému testu' : 'Doplňte OPENAI_API_KEY na serveru';
    $('directory-date').textContent = config.directory_fetched_at ? `Adresář načten ${new Date(config.directory_fetched_at).toLocaleString('cs-CZ')}` : 'Adresář zatím není načten. Spusťte import podle README.';
    $('start-voice').disabled = !config.voice_ready;
    $('demo').disabled = false;
    $('dial').disabled = !config.outbound_enabled;
    await refreshHistory();
    if (config.voice_ready) prepareGreeting().catch(() => {});
  } catch (error) {
    $('connection').textContent = 'Server nedostupný';
    message(error.message);
  }
}

$('demo').addEventListener('click',async()=>{setBusy();try{render(await api('/demo',{method:'POST'}));renderCaptions([{type:'session.input_transcript.delta',delta:'[Syntetická ukázka] Vybrala jsem si pracoviště v Brně. Vyhovuje mi odpoledne. Ano, potvrďte a poznamenejte si tento ukázkový termín.'},{type:'session.output_transcript.delta',delta:'Máme to domluvené. Termín i pracoviště najdete v přehledu.'}]);await refreshHistory();}catch(e){message(e.message);}finally{cleanup();}});
$('start-voice').addEventListener('click',async()=>{setBusy();captions=[];let localId=null;try{await greeting.unlock();greeting.arm();await prepareGreeting();media=await navigator.mediaDevices.getUserMedia(inputConstraints($('audio-input').value, $('virtual-input').checked));const connection=new RTCPeerConnection();peer=connection;connection.onconnectionstatechange=()=>{if(connection.connectionState==='connected')greeting.start();};connection.ontrack=ev=>{$('audio').hidden=false;$('audio').srcObject=new MediaStream([ev.track]);$('audio').play().catch(()=>message('Pro poslech stiskněte přehrávání v ovladači zvuku.'));};for(const track of media.getAudioTracks())connection.addTrack(track,media);channel=connection.createDataChannel('oai-events');channel.onmessage=({data})=>{try{onEvent(JSON.parse(data));}catch{message('Nepodařilo se zpracovat událost hlasové služby.');}};channel.onclose=()=>{if(active){message('Zvukové spojení bylo přerušeno. Ověřte ukončení hovoru.');cleanup();refreshCurrent();}};await connection.setLocalDescription(await connection.createOffer());await gatherIce(connection);current=await api('/sessions',{method:'POST',body:JSON.stringify({mode:'browser',contact_label:'Hlasový test',...speechPreferences()})});localId=current.id;render(current);const result=await api(`/sessions/${current.id}/offer`,{method:'POST',body:JSON.stringify({sdp:connection.localDescription.sdp})});await connection.setRemoteDescription({type:'answer',sdp:result.transport.sdp});active=true;connecting=false;document.body.classList.add('live');$('stop').disabled=false;startPoll();}catch(e){if(localId)await api(`/sessions/${localId}/stop`,{method:'POST'}).catch(()=>{});cleanup();message(e.message);}});
$('stop').addEventListener('click',async()=>{if(!current)return;$('stop').disabled=true;try{if(channel?.readyState==='open'){channel.send(JSON.stringify({type:'session.close'}));closeTimer=setTimeout(async()=>{await api(`/sessions/${current.id}/stop`,{method:'POST'}).catch(e=>message(e.message));cleanup();await refreshCurrent();},10000);}else{await api(`/sessions/${current.id}/stop`,{method:'POST'});setTimeout(async()=>{cleanup();await refreshCurrent();await refreshHistory();},1000);}}catch(e){message(e.message);$('stop').disabled=false;}});
$('dial').addEventListener('click',async()=>{if(!$('permission').checked)return message('Potvrďte oprávnění ke kontaktování této osoby.');const phone=$('phone').value.trim();if(!/^\+[1-9]\d{7,14}$/.test(phone))return message('Zadejte číslo v mezinárodním formátu, například +420…');setBusy();try{current=await api('/sessions',{method:'POST',body:JSON.stringify({mode:'phone',contact_label:'Telefonní rozhovor',...speechPreferences()})});render(current);await api(`/sessions/${current.id}/dial`,{method:'POST',body:JSON.stringify({phone,contact_permission_confirmed:true})});$('phone').value='';$('permission').checked=false;active=true;connecting=false;$('stop').disabled=false;document.body.classList.add('live');startPoll();}catch(e){cleanup();message(e.message);await refreshCurrent();}});
let searchTimer;let searchVersion=0;$('search').addEventListener('input',()=>{clearTimeout(searchTimer);const version=++searchVersion;searchTimer=setTimeout(async()=>{try{const result=await api(`/centers?q=${encodeURIComponent($('search').value)}`);if(version!==searchVersion)return;const target=$('centers');target.replaceChildren();for(const c of result.centers){const card=node('article',undefined,'center-card');card.append(node('h3',c.name),node('p',`${c.address}, ${c.postal_code} ${c.city}`),node('p',c.phone.join(' · ')));if(c.distance_km!==undefined)card.append(node('p',`Asi ${c.distance_km.toLocaleString('cs-CZ')} km vzdušnou čarou od středu obce`,'wait'));if(c.reported_wait_days!==null)card.append(node('p',`Orientační čekání: ${c.reported_wait_days} dní · údaj ${c.wait_updated||'bez data'}`,'wait'));card.append(safeLink('Detail pracoviště ↗',c.source_url));target.append(card);}if(!result.centers.length)target.append(node('p',result.needs_clarification?'Název místa je nejednoznačný. Doplňte okres nebo blízké větší město.':'Bez shody. Zkuste název centra nebo blízké větší město.','empty'));}catch(e){message(e.message);}},250);});
$('refresh-history').addEventListener('click',()=>refreshHistory().catch(e=>message(e.message)));
$('export').addEventListener('click',()=>{if(!current)return;const {captions:ignored,phone_hash:phoneHash,...record}=current;const blob=new Blob([JSON.stringify(record,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`mamo-${current.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
$('delete').addEventListener('click',async()=>{if(!current)return;try{await api(`/sessions/${current.id}`,{method:'DELETE'});current=null;$('outcome').replaceChildren(node('p','Záznam byl smazán.','empty'));$('record-actions').hidden=true;$('captions').replaceChildren();await refreshHistory();}catch(e){message(e.message);}});
window.addEventListener('pagehide',()=>{media?.getTracks().forEach(t=>t.stop());peer?.close();if(current&&active)fetch(`/api/sessions/${current.id}/stop`,{method:'POST',keepalive:true}).catch(()=>{});});

$('refresh-devices').addEventListener('click',async()=>{if(active||connecting)return message('Nejdříve ukončete asistentku.');try{const previous=$('audio-input').value||localStorage.getItem('mamo-audio-input');const inputs=await availableInputs(navigator.mediaDevices);const select=$('audio-input');select.replaceChildren(new Option('Výchozí mikrofon',''));for(const device of inputs)select.add(new Option(device.label||'Mikrofon',device.deviceId));if(inputs.some(d=>d.deviceId===previous))select.value=previous;message();}catch(e){message(e.message);}});

$('audio-input').addEventListener('change',()=>localStorage.setItem('mamo-audio-input',$('audio-input').value));
$('virtual-input').checked=localStorage.getItem('mamo-virtual-input')==='true';
$('virtual-input').addEventListener('change',()=>localStorage.setItem('mamo-virtual-input',String($('virtual-input').checked)));

initialize();
