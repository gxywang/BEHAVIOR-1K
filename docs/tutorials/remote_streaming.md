# :material-monitor: **Remote Streaming through WebRTC**

Run OmniGibson on a remote server and watch its viewport on your own machine.

## Enabling remote streaming

```{.shell .annotate}
export OMNIGIBSON_REMOTE_STREAMING=webrtc
```

Do **not** also set `OMNIGIBSON_HEADLESS=1`. Streaming already launches the application windowless, and code that
branches on `gm.HEADLESS` — including OmniGibson's own viewport-camera setup — will otherwise skip work you need,
so you end up streaming a default camera pose.

On startup the log prints the address to connect to.

## Connecting

Install the standalone [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
and enter the server's address — a bare hostname or IP, with no port. The client supplies the port itself.

If the server is not directly reachable, forward the port over SSH and connect the client to `127.0.0.1`:

```{.shell .annotate}
ssh -N -L 49100:127.0.0.1:49100 <user>@<server>
```

An SSH tunnel is enough because OmniGibson selects the WebSocket transport (`/app/livestream/proto = "websocket"`),
so the video rides the same TCP connection as the signalling. No UDP media port is opened — which is what usually
makes WebRTC impossible to tunnel.

## Ports

Only **TCP 49100** has to reach the server.

`OMNIGIBSON_WEBRTC_PORT` changes the port the server listens on, but the desktop client always dials 49100, so
changing it means you must forward 49100 on the client side to whatever you chose. Leaving it alone is simplest.

`OMNIGIBSON_PUBLIC_IP` sets the address the server advertises to the client. Set it when the machine's outward
address differs from the one it detects for itself, e.g. behind NAT or with several interfaces.

!!! warning "`OMNIGIBSON_HTTP_PORT` does nothing on Isaac Sim 5.x"

    Earlier releases served a browser-based client on port 8211. The extension behind it
    (`omni.services.streamclient.webrtc`) is no longer shipped, nothing binds the port, and there is no browser
    client — the desktop application above is the only way in. The variable is kept for backwards compatibility.

## Troubleshooting

**Nothing but the scene should appear.** Auxiliary cameras — robot-mounted sensors, extra `VisionSensor`s — do not
get their own windows composited into the stream while `OMNIGIBSON_REMOTE_STREAMING` is set. If you see extra
camera tiles, check that your OmniGibson is recent enough to include that behaviour.

**Read the log when a session misbehaves.** Streaming diagnostics are enabled (`/app/livestream/logLevel` and the
WebRTC QoS callback), so the Kit log records signalling, peer state and disconnect reason codes rather than a bare
connect/disconnect. Look for `carb.livestream-rtc.plugin` lines in
`<appdata>/local/logs/Kit/OmniGibson/<version>/kit_*.log`.

**A connected session with no picture** usually means the client attached before the renderer produced its first
frame. Reconnecting the client is the quickest check.
