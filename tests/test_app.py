import io
from pathlib import Path
from PIL import Image
from app import create_app

def make_image():
    data=io.BytesIO(); Image.new('RGB',(180,180),(80,150,60)).save(data,'PNG');data.seek(0);return data

def app(tmp_path):
    return create_app({'TESTING':True,'DATABASE':str(tmp_path/'test.db'),'UPLOAD_FOLDER':str(tmp_path/'uploads'),'SECRET_KEY':'test'})

def test_pages_load(tmp_path):
    client=app(tmp_path).test_client()
    for path in ('/','/scan','/guide','/history','/about'): assert client.get(path).status_code==200

def test_upload_creates_result(tmp_path):
    client=app(tmp_path).test_client(); response=client.post('/scan',data={'image':(make_image(),'leaf.png')},content_type='multipart/form-data')
    assert response.status_code==302 and '/result/' in response.headers['Location']

def test_bad_extension_is_rejected(tmp_path):
    client=app(tmp_path).test_client(); response=client.post('/scan',data={'image':(io.BytesIO(b'not image'),'leaf.txt')},content_type='multipart/form-data',follow_redirects=True)
    assert b'JPG, JPEG, PNG, or WEBP' in response.data
