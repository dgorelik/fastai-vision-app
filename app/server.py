from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

from google.cloud import storage

model_file_name = 'export.pkl'
bucket_name = 'dg-storage-bucket'
classes = ['amtrak train', 'british train']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

def set_credentials():
    import os
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=str(path/"gcp_cred.json")

async def download_file(dest):
    if dest.exists(): return
    set_credentials()
    storage_client =  storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(model_file_name)
    blob.download_to_filename(str(dest))

async def setup_learner():
    await download_file(path/'models'/f'{model_file_name}')
    learn = load_learner(path/'models')
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    result, _, probs = learn.predict(img)
    result_text = '{} with probability {}'.format(result, max(probs))
    return JSONResponse({'result': result_text})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

