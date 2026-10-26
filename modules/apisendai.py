
import os
import base64
import mimetypes
from openai import OpenAI


class sendai:
    api_base = "https://api.deepseek.com"

    def __init__(self, api_key, temp: float = 1.0, tokens: int = 700):
        if not api_key:
            raise ValueError("An API key must be provided")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.api_base
        )
        self.temp = temp
        self.tokens = tokens

    def sendmessage(self, rolecontent, usercontent) -> str:
        response = self.client.chat.completions.create(
            model="deepseek-flash",
            messages=[
                {"role": "system", "content": rolecontent},
                {"role": "user", "content": usercontent}
            ],
            temperature=self.temp,
            max_tokens=self.tokens
        )

        return response.choices[0].message.content

    def imggguess(self, img):
        # img is a Flask FileStorage object
        image_data = base64.b64encode(
            img.read()
        ).decode("utf-8")

        print(img)

        mime_type = img.mimetype or "image/jpeg"

        prompt = (
                """Identify the plant. Only tell me its common name with single word.
                Sample: 'Plant name: {plantname}' """)

        response = self.client.chat.completions.create(
            model="deepseek-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{mime_type};base64,"
                                    f"{image_data}"
                                )
                            }
                        }
                    ]
                }
            ]
            #max_tokens=380
        )
        print(response.choices[0].message)

        return response.choices[0].message.content.strip()
