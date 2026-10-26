import requests


class sendpost:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

        if not api_key:
            raise ValueError("any api_key not provided")

    def cppost(self, message: str, mediaid=None) -> str:

        api_url = "https://mastodon.social/api/v1/statuses"

        message = {"status": f" {message} "}
        if mediaid:
            message["media_ids[]"] = [mediaid]


        link = requests.post(api_url, headers=self.headers, data=message)

        return link

    def imgupload(self, img):
        api_url = "https://mastodon.social/api/v1/media"

        file = {"file": (img.filename, img.stream, img.mimetype)}

        pst = requests.post(api_url, headers=self.headers, files=file)
        theid = pst.json()

        return theid.get("id")


