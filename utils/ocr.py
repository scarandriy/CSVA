import easyocr
reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext('scam.png', detail=0)
print(result)