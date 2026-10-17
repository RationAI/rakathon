import assert from 'node:assert/strict';
import {test} from 'node:test';
import {GreetingPlayer} from '../static/greeting.js';

function fixture() {
  const sources = [];
  let fetches = 0;
  const player = new GreetingPlayer({createContext: () => ({
    resume: async () => {}, destination: {}, decodeAudioData: async bytes => bytes.byteLength,
    createBufferSource: () => {
      const source = {starts: 0, stops: 0, connect() {}, disconnect() {},
        start() {this.starts++;}, stop() {this.stops++; this.onended();}};
      sources.push(source); return source;
    }
  })});
  return {player, sources, fetchClip: async () => {fetches++; return new ArrayBuffer(1);}, fetchCount: () => fetches};
}

test('greeting starts on connection with no caller speech, only once, and reuses the clip', async () => {
  const f = fixture();
  await f.player.unlock();
  await f.player.prepare(f.fetchClip);
  f.player.arm(); f.player.start(); f.player.start();
  assert.equal(f.sources.length, 1);
  assert.equal(f.sources[0].starts, 1);
  f.player.cancel();
  assert.equal(f.sources[0].stops, 1);
  f.player.arm(); await f.player.prepare(f.fetchClip); f.player.start();
  assert.equal(f.sources.length, 2);
  assert.equal(f.fetchCount(), 1);
});

test('caller speech cancels a queued or playing greeting', async () => {
  const f = fixture();
  await f.player.prepare(f.fetchClip);
  f.player.arm(); f.player.cancel(); f.player.start();
  assert.equal(f.sources.length, 0);
  f.player.arm(); f.player.start(); f.player.cancel();
  assert.equal(f.sources[0].stops, 1);
});

test('failed greeting preparation can be retried', async () => {
  const f = fixture();
  await assert.rejects(f.player.prepare(async () => {throw new Error('unavailable');}));
  await f.player.prepare(f.fetchClip);
  f.player.arm(); f.player.start();
  assert.equal(f.sources[0].starts, 1);
});

test('changing voice or tone uses the matching clip, even when an older request finishes late', async () => {
  const f = fixture();
  let finishOld;
  const old = f.player.prepare(() => new Promise(resolve => {finishOld = resolve;}), 'coral-positive');
  await f.player.prepare(async () => new ArrayBuffer(2), 'shimmer-gentle');
  assert.equal(f.player.buffer, 2);
  finishOld(new ArrayBuffer(1));
  await old;
  assert.equal(f.player.buffer, 2);
  await f.player.prepare(async () => {throw new Error('must reuse cached clip');}, 'coral-positive');
  assert.equal(f.player.buffer, 1);
});
