export function inputConstraints(deviceId, virtualInput) {
  const constraints = {
    echoCancellation: !virtualInput,
    noiseSuppression: !virtualInput,
    autoGainControl: !virtualInput,
  };
  if (deviceId) constraints.deviceId = {exact: deviceId};
  return {audio: constraints, video: false};
}

export async function availableInputs(mediaDevices) {
  let permissionStream;
  try {
    permissionStream = await mediaDevices.getUserMedia({audio: true, video: false});
    return (await mediaDevices.enumerateDevices()).filter(device => device.kind === 'audioinput');
  } finally {
    permissionStream?.getTracks().forEach(track => track.stop());
  }
}
