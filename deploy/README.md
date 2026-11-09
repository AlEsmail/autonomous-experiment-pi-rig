# Deploying to the Pi (`<pi-hostname>`)

One-time setup on the Pi:

```bash
cd ~
git clone https://github.com/AlEsmail/<public-repo-name>.git   # public: no authentication needed
cd pantone-rig
python3 -m venv --system-site-packages .venv            # system site gives gpiozero + lgpio from Pi OS
.venv/bin/pip install fastapi uvicorn httpx pytest anthropic
sudo cp deploy/rig-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rig-service

sudo cp deploy/scientist.service /etc/systemd/system/
cp deploy/scientist.env.example deploy/scientist.env
chmod 600 deploy/scientist.env
# edit deploy/scientist.env and replace paste-key-here with the real Anthropic API key
sudo systemctl daemon-reload
sudo systemctl enable --now scientist
```

Each deploy:

```bash
cd ~/pantone-rig && git pull && sudo systemctl restart rig-service
```

Close any open `/wall` or `/operate` tabs before a restart or stop. An open MJPEG stream
(`/stream.mjpg`) holds its connection open until uvicorn's graceful-shutdown timeout expires
(5 s, set in `src/rig_service/__main__.py`), so the stop can take a few seconds; a mid-flight
camera capture can add a couple more, capped by the driver timeout. With no stream open the stop
is immediate. The bounded wait is by design, not a fault.

Verify:

```bash
curl http://localhost:8000/status
curl -X POST http://localhost:8000/pump -H 'Content-Type: application/json' -d '{"channel":"A","ml":2}'
```

Verify the scientist:

```bash
sudo systemctl status scientist
sudo journalctl -u scientist -f
```

When idle, the scientist polls rig-service slowly and does not call the Anthropic API until
a run is active **and** its mode is set to `auto`.

Boot-safety check across a reboot: `sudo reboot`, bank must stay silent through boot,
service comes back via systemd and re-arms all relays off before serving.

## Remote operation (Tailscale)

Both ends are on the tailnet. Use the MagicDNS name —
it works from anywhere and is more reliable than `<pi-hostname>.local` mDNS:

- SSH: `ssh <user>@<pi-hostname>`
- Rig API: `http://<pi-hostname>:8000/status`
- Tailscale IPs: Pi `<tailscale-ip>`, laptop `<tailscale-ip>`

## Getting code onto the Pi

The laptop pushes directly over SSH (remote `pi`, works over LAN or Tailscale):

```bash
git push pi main    # Pi repo has receive.denyCurrentBranch=updateInstead
ssh <user>@<pi-hostname> "sudo systemctl restart rig-service"
```

`git pull` from GitHub on the Pi needs the Pi's deploy key added to the repo
(`~/.ssh/id_ed25519.pub` on the Pi → GitHub repo settings → Deploy keys, read-only).
