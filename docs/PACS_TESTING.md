# PACS Testing with Orthanc

This document describes how to set up a local Orthanc PACS server for testing Sauhu's DICOM networking features.

## Prerequisites

- Docker installed and running
- Test DICOM images (e.g., from `testdata/` directory)

## 1. Orthanc Configuration

Create the configuration file `/tmp/orthanc-config/orthanc.json`:

```json
{
  "Name" : "Orthanc",
  "DicomAet" : "ORTHANC",
  "DicomPort" : 4242,
  "HttpPort" : 8042,
  "RemoteAccessAllowed" : true,
  "AuthenticationEnabled" : false,
  "DicomModalities" : {
    "SAUHU" : {
      "AET" : "SAUHU",
      "Host" : "172.17.0.1",
      "Port" : 11112,
      "AllowEcho" : true,
      "AllowFind" : true,
      "AllowMove" : true,
      "AllowStore" : true
    }
  },
  "DicomAlwaysAllowStore" : true,
  "DicomCheckCalledAet" : false
}
```

**Important notes:**
- `172.17.0.1` is the Docker bridge gateway IP (host machine from container's perspective)
- Port 11112 is Sauhu's default SCP port
- `DicomAlwaysAllowStore` allows any AET to store images to Orthanc

## 2. Start Orthanc

```bash
mkdir -p /tmp/orthanc-config /tmp/orthanc-db

# Create config file (see above)
cat > /tmp/orthanc-config/orthanc.json << 'EOF'
{
  "Name" : "Orthanc",
  "DicomAet" : "ORTHANC",
  "DicomPort" : 4242,
  "HttpPort" : 8042,
  "RemoteAccessAllowed" : true,
  "AuthenticationEnabled" : false,
  "DicomModalities" : {
    "SAUHU" : {
      "AET" : "SAUHU",
      "Host" : "172.17.0.1",
      "Port" : 11112,
      "AllowEcho" : true,
      "AllowFind" : true,
      "AllowMove" : true,
      "AllowStore" : true
    }
  },
  "DicomAlwaysAllowStore" : true,
  "DicomCheckCalledAet" : false
}
EOF

# Start Orthanc container
docker run -d --name orthanc \
  -p 4242:4242 \
  -p 8042:8042 \
  -v /tmp/orthanc-config:/etc/orthanc:ro \
  -v /tmp/orthanc-db:/var/lib/orthanc/db \
  orthancteam/orthanc

# View logs
docker logs -f orthanc
```

## 3. Firewall Configuration

The host firewall must allow Docker to connect back to Sauhu's SCP:

```bash
# Allow Docker network (172.17.0.0/16) to access port 11112
sudo ufw allow from 172.17.0.0/16 to any port 11112 proto tcp
```

## 4. Sauhu Configuration

Create `~/.config/sauhu/config.toml`:

```toml
[local]
ae_title = "SAUHU"
port = 11112
storage_path = "~/.cache/sauhu/pacs"

[pacs.servers.orthanc]
name = "Orthanc Local"
ae_title = "ORTHANC"
host = "localhost"
port = 4242
```

## 5. Upload Test Images to Orthanc

### Using C-STORE (DICOM)

```bash
# Upload a single file
cargo run --release -- pacs store testdata/CT_small.dcm

# Upload multiple files
cargo run --release -- pacs store testdata/CT_small.dcm testdata/emri_small.dcm
```

### Using REST API (alternative)

```bash
curl -X POST http://localhost:8042/instances \
  --data-binary @testdata/CT_small.dcm
```

## 6. Test DICOM Operations

### C-FIND (Query)

```bash
# Find all studies
cargo run --release -- pacs find

# Find by patient ID
cargo run --release -- pacs find --patient-id "1CT1"

# Find by patient name
cargo run --release -- pacs find --patient-name "*Smith*"
```

### C-MOVE (Retrieve)

```bash
# Retrieve by patient ID
cargo run --release -- pacs move --patient-id "1CT1" --dest SAUHU

# Retrieve by study UID
cargo run --release -- pacs move --study-uid "1.2.3.4.5" --dest SAUHU
```

Retrieved files are saved to `~/.cache/sauhu/pacs/<study_uid>/`.

## 7. Verify from GUI

1. Start Sauhu: `cargo run --release`
2. Press `D` to open Database window
3. Click "Query PACS" to search for studies
4. Double-click a study to retrieve and open it

## Troubleshooting

### C-MOVE fails with 0xC000

1. **Check firewall**: Verify Orthanc can reach Sauhu's SCP port
   ```bash
   docker run --rm alpine nc -zv 172.17.0.1 11112 -w 5
   ```

2. **Check Orthanc config**: Ensure SAUHU is in DicomModalities
   ```bash
   docker logs orthanc 2>&1 | grep -i sauhu
   ```

3. **Check SCP is running**: Verify port is listening
   ```bash
   ss -tlnp | grep 11112
   ```

### Connection timeout

- Ensure Docker is using bridge network (default)
- Verify `172.17.0.1` is the correct gateway:
  ```bash
  docker network inspect bridge | grep Gateway
  ```

### Permission denied

- Check file permissions on storage directory
- Ensure firewall rule is added with sudo

## Cleanup

```bash
docker stop orthanc && docker rm orthanc
rm -rf /tmp/orthanc-config /tmp/orthanc-db
```
