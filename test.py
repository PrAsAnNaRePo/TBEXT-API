import requests

def test_categorize():
    url = "http://0.0.0.0:8000/categorize"
    pdf_file_path = "./P31-(1) BALLAST WATER TREATMENT SYSTEM.pdf"
    selected_pages = '201,202'

    with open(pdf_file_path, "rb") as f:
        files = {"pdf_file": ("file.pdf", f, "application/pdf")}
        data = {"selected_pages": selected_pages}

        response = requests.post(url, files=files, data=data)

    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

    assert response.status_code == 200
    assert isinstance(response.json(), list)  # Expecting a list of results
    for result in response.json():
        assert "page_num" in result
        assert "category" in result
        assert "bbox" in result
        assert "dpi" in result
        assert "img_height" in result
        assert "img_width" in result

if __name__ == "__main__":
    test_categorize()
