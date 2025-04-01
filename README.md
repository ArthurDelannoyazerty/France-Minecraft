```bash
uv venv -p 3.13
source .venv/Scripts/activate
uv pip compile requirements.in -o requirements.txt --no-header --no-annotate 

```