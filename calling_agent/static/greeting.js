// A cached, application-played greeting starts without waiting for caller audio.
export class GreetingPlayer {
  constructor({onStart = () => {}, onEnd = () => {}, createContext = () => new AudioContext()} = {}) {
    this.onStart = onStart;
    this.onEnd = onEnd;
    this.createContext = createContext;
    this.context = null;
    this.prepared = null;
    this.cache = new Map();
    this.key = null;
    this.buffer = null;
    this.source = null;
    this.armed = false;
    this.started = false;
  }
  unlock() {
    this.context ??= this.createContext();
    return this.context.resume();
  }
  prepare(fetchClip, key = 'default') {
    this.context ??= this.createContext();
    if (this.key !== key) { this.key = key; this.buffer = null; }
    if (!this.cache.has(key)) {
      const pending = fetchClip().then(bytes => this.context.decodeAudioData(bytes)).catch(error => {
        this.cache.delete(key);
        throw error;
      });
      this.cache.set(key, pending);
    }
    this.prepared = this.cache.get(key).then(buffer => {
      if (this.key === key) this.buffer = buffer;
    });
    return this.prepared;
  }
  arm() { this.cancel(); this.armed = true; this.started = false; }
  start() {
    if (!this.armed || this.started || !this.buffer) return;
    this.started = true;
    const source = this.context.createBufferSource();
    this.source = source;
    source.buffer = this.buffer;
    source.connect(this.context.destination);
    source.onended = () => {
      if (this.source === source) { this.source = null; this.onEnd(); }
      source.disconnect();
    };
    this.onStart();
    source.start();
  }
  cancel() {
    this.armed = false;
    if (this.source) {
      const source = this.source;
      this.source = null;
      source.stop();
      this.onEnd();
    }
  }
}
