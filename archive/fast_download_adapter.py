#!/usr/bin/env python3
"""
fast_download_adapter.py
Split adapter_model.safetensors into N chunks on server,
download all chunks + small files in parallel, then reassemble.
Run AFTER training completes.
"""
import subprocess, os, sys, time, threading

SERVER   = "ubuntu@98.84.34.154"
KEY      = os.path.expanduser("~/Downloads/qlora.pem")
REMOTE   = "~/memory-policy/adapter/memory-policy-dpo-v3"
LOCAL    = "/Users/saichandavula/Documents/memory-policy/adapter/memory-policy-dpo-v3"
CHUNKS   = 4   # parallel connections

SSH = ["ssh", "-i", KEY, SERVER]
SCP = ["scp", "-i", KEY, "-o", "Compression=no"]

def ssh(cmd):
    return subprocess.run(SSH + [cmd], capture_output=True, text=True, check=True).stdout.strip()

def wait_for_training():
    print("⏳ Waiting for DPO v3 training to complete...")
    while True:
        try:
            out = ssh("pgrep -f dpo_training_v3 && echo RUNNING || echo DONE")
            if "DONE" in out:
                print("✅ Training complete!")
                break
        except:
            pass
        time.sleep(30)

def split_on_server(chunk_n):
    """Split safetensors into N parts on server."""
    print(f"📦 Splitting adapter into {chunk_n} parts on server...")
    ssh(f"cd ~/memory-policy && python3 -c \"\
import os; \
f = open('{REMOTE.replace('~/', '')}/adapter_model.safetensors','rb'); \
data = f.read(); f.close(); \
size = len(data); \
chunk = (size + {chunk_n}-1) // {chunk_n}; \
[open(f'adapter_v3_chunk_{{i:02d}}','wb').write(data[i*chunk:min((i+1)*chunk,size)]) for i in range({chunk_n})]; \
print(f'Split {{size}} bytes into {chunk_n} chunks of ~{{chunk}} bytes'); \
os.system('ls -lh adapter_v3_chunk_*') \
\" ")
    print("✅ Server-side split done")

def download_chunk(i, results):
    """Download one chunk in background."""
    remote_file = f"{SERVER}:~/memory-policy/adapter_v3_chunk_{i:02d}"
    local_file  = f"{LOCAL}/adapter_v3_chunk_{i:02d}"
    t0 = time.time()
    r = subprocess.run(SCP + [remote_file, local_file], capture_output=True, text=True)
    elapsed = time.time() - t0
    results[i] = (r.returncode == 0, elapsed)
    status = "✅" if r.returncode == 0 else "❌"
    print(f"  {status} Chunk {i:02d} done in {elapsed:.0f}s")

def download_small_files():
    """Download all small files at once."""
    small = ["adapter_config.json","tokenizer.json","tokenizer_config.json",
             "chat_template.jinja","README.md","special_tokens_map.json",
             "vocab.json","merges.txt","added_tokens.json"]
    for f in small:
        try:
            subprocess.run(SCP + [f"{SERVER}:{REMOTE}/{f}", f"{LOCAL}/"],
                           capture_output=True, timeout=30)
        except:
            pass
    print("✅ Small files downloaded")

def reassemble(chunk_n):
    """Cat all chunks back into adapter_model.safetensors."""
    print("🔧 Reassembling chunks...")
    chunk_files = " ".join(f"{LOCAL}/adapter_v3_chunk_{i:02d}" for i in range(chunk_n))
    subprocess.run(f"cat {chunk_files} > {LOCAL}/adapter_model.safetensors", shell=True, check=True)
    # Cleanup chunks
    subprocess.run(f"rm -f {LOCAL}/adapter_v3_chunk_*", shell=True)
    # Also cleanup on server
    ssh(f"rm -f ~/memory-policy/adapter_v3_chunk_*")
    size_mb = os.path.getsize(f"{LOCAL}/adapter_model.safetensors") / 1024 / 1024
    print(f"✅ Reassembled → adapter_model.safetensors ({size_mb:.1f} MB)")

def verify():
    files = os.listdir(LOCAL)
    print(f"\n📁 adapter/memory-policy-dpo-v3/")
    for f in sorted(files):
        sz = os.path.getsize(f"{LOCAL}/{f}")
        print(f"   {f:<40} {sz/1024/1024:.2f} MB")

def main():
    os.makedirs(LOCAL, exist_ok=True)

    # 1. Wait for training
    wait_for_training()
    time.sleep(5)

    # 2. Split on server
    split_on_server(CHUNKS)

    # 3. Download in parallel
    t_start = time.time()
    print(f"\n⬇️  Downloading {CHUNKS} chunks + small files in parallel...")
    results = {}
    threads = []

    # Chunk download threads
    for i in range(CHUNKS):
        t = threading.Thread(target=download_chunk, args=(i, results))
        t.start(); threads.append(t)

    # Small files in parallel thread
    sf = threading.Thread(target=download_small_files)
    sf.start(); threads.append(sf)

    for t in threads:
        t.join()

    elapsed = time.time() - t_start
    success = all(v[0] for v in results.values())
    if not success:
        print("❌ Some chunks failed!")
        sys.exit(1)

    print(f"\n⚡ All {CHUNKS} chunks downloaded in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # 4. Reassemble
    reassemble(CHUNKS)

    # 5. Verify
    verify()
    print(f"\n🎉 DPO v3 adapter ready at adapter/memory-policy-dpo-v3/")

if __name__ == "__main__":
    main()
