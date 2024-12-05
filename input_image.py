import requests

# 서버의 URL
url = 'http://localhost:5000/detect_visual'

# 이미지 파일 업로드
with open("C:/users/hjm66/RedSWUS/RedSWUS-flask/tf_project/TextFuseNet_ed/input/input.jpg", "rb") as f:
    files = {'image': f}
    response = requests.post(url, files=files)

# 응답으로 받은 이미지 저장
if response.status_code == 200:
    with open("C:/Users/hjm66/RedSWUS/RedSWUS-flask/tf_project/TextFuseNet_ed/output/output.jpg", "wb") as f:
        f.write(response.content)
    print("처리된 이미지가 저장되었습니다: C:/Users/hjm66/RedSWUS/RedSWUS-flask/tf_project/TextFuseNet_ed/output/output.jpg")
else:
    print("Error:", response.status_code, response.text)