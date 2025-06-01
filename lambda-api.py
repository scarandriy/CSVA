import base64
from openai import OpenAI
from core.logger                import EvaluationLogger

# Set API credentials and endpoint
openai_api_key = "---"
openai_api_base = "https://api.lambda.ai/v1"
model = "llama-4-maverick-17b-128e-instruct-fp8"
TEST3 = """
You are acting as a professional scam and security threat detection system. 
You are analyzing a screenshot from a user's screen. Your task is to evaluate whether the current screenshot contains any indications of scams, fraud, or malicious behavior to assist your judgment.

Final Answer Format:
{  
"reasoning": "<...>",
"scam_indicators": "[indicators]"
"scam_type": "<type of scam, if applicable>",
"Scam Probability": <integer from 0 to 5>
}  
"""

TEST2 = """
You are acting as a professional scam and security threat detection system. 
You are analyzing a screenshot from a user's screen. Your task is to evaluate whether the current screenshot contains any indications of scams, fraud, or malicious behavior to assist your judgment.

LIST POSSIBLE SCAM INDICATORS:

Email Scam Indicators:
- Suspicious Sender: Sender address mismatch or unknown
- Unusual Links: Redirects to dubious or unexpected sites
- Sense of Urgency: Pressures for immediate action
- Lack of Personalization: Generic greetings or sign-offs

Website Scam Indicators:
- URL mismatch: URL doesn't match with the web site
- Suspicious URL: unusual patterns, attempts to seem like a trusted URL
- Low Prices / Free Claims: Unrealistically low prices or freebies
- High Payout Promises: Promises of unusually large sums
- Non-existent Company Info: Fake or ambiguous company details
- Branding Inconsistencies: Inconsistent logos or color schemes
- Image Manipulation: Signs of tampering or low-quality images
- Text-Image Mismatch: Discrepancies between text and visuals
- UI/UX Irregularities: Unusual or mismatched interface elements

Messenger Scam Indicators:
- Unsolicited Messaging: Generic messages from unknown contacts
- Urgent Financial Requests: Demands for immediate transfers
- Sensitive Info Requests: Asks for personal or sensitive details

Interface Scam Indicators:
- Confirmshaming: Guilt-tripping users into making a specific choice
- Fake Scarcity / Fake Urgency: Creating false scarcity or urgency (“Only 3 left in stock”)
- Nudge: Emphasizing a preferred option visually while de-emphasizing others
- Obstruction: Designing unnecessary hurdles to prevent opting out or cancelling
- Forced-Action: Requiring completion of an unrelated action before proceeding
- Pre-Selection: Defaulting checkboxes or options to “on”
- Visual Interference: Hiding critical information via low contrast or small text
- Jargon: Using complex or misleading language to confuse users
- Hidden Subscription: Not clearly stating that signup enrolls in a service
- Sneaking Hidden Costs: Revealing extra fees only at final checkout
- Disguised Ads: Embedding ads disguised as normal UI elements
- Trickwording: Phrasing that misleads or reverses opt-in and opt-out actions

Scam Probability Scale:
- 0 - No scam signs: Trusted, verified source; no signs of scam.
- 1 - Uncertain: No immediate red flags, but not enough data to confirm safety.
- 2 - Low scam probability: Weak anomalies; likely safe, but caution advised.
- 3 - Moderate scam probability: Suggests possible scam/fraud. Review is needed. ANY popups about prizes, rewards, or milestones MUST be at least this level.
- 4 - High scam probability: Strong indicators of malicious or scam-related activity.
- 5 - Critical scam probability: Confirmed scam, malware, or phishing attempt. Urgent action required.


Answer Format:
{  
"reasoning": "<...>",
"scam_indicators": "[indicators]"
"scam_type": "<type of scam, if applicable>",
"Scam Probability": <integer from 0 to 5>
}  
"""

TEST1 = """
You are acting as a professional scam and security threat detection system. 
You are analyzing a screenshot from a user's screen. Your task is to evaluate whether the current screenshot contains any indications of scams, fraud, or malicious behavior to assist your judgment.

LIST POSSIBLE SCAM INDICATORS:

Email Scam Indicators:
- Suspicious Sender: Sender address mismatch or unknown
- Unusual Links: Redirects to dubious or unexpected sites
- Sense of Urgency: Pressures for immediate action
- Lack of Personalization: Generic greetings or sign-offs

Website Scam Indicators:
- URL mismatch: URL doesn't match with the web site
- Suspicious URL: unusual patterns, attempts to seem like a trusted URL
- Low Prices / Free Claims: Unrealistically low prices or freebies
- High Payout Promises: Promises of unusually large sums
- Non-existent Company Info: Fake or ambiguous company details
- Branding Inconsistencies: Inconsistent logos or color schemes
- Image Manipulation: Signs of tampering or low-quality images
- Text-Image Mismatch: Discrepancies between text and visuals
- UI/UX Irregularities: Unusual or mismatched interface elements

Messenger Scam Indicators:
- Unsolicited Messaging: Generic messages from unknown contacts
- Urgent Financial Requests: Demands for immediate transfers
- Sensitive Info Requests: Asks for personal or sensitive details

Interface Scam Indicators:
- Confirmshaming: Guilt-tripping users into making a specific choice
- Fake Scarcity / Fake Urgency: Creating false scarcity or urgency (“Only 3 left in stock”)
- Nudge: Emphasizing a preferred option visually while de-emphasizing others
- Obstruction: Designing unnecessary hurdles to prevent opting out or cancelling
- Forced-Action: Requiring completion of an unrelated action before proceeding
- Pre-Selection: Defaulting checkboxes or options to “on”
- Visual Interference: Hiding critical information via low contrast or small text
- Jargon: Using complex or misleading language to confuse users
- Hidden Subscription: Not clearly stating that signup enrolls in a service
- Sneaking Hidden Costs: Revealing extra fees only at final checkout
- Disguised Ads: Embedding ads disguised as normal UI elements
- Trickwording: Phrasing that misleads or reverses opt-in and opt-out actions


Tools Available:
- Image-to-Text (input -- image): A tool which produces a detailed textual representation of a given image.
- Verify URL (input -- URL string): A tool which analyzes a URL of a web-page. 
- IsMessenger (input -- textual representation of an image): A tool which tells, whether a current open window is a messenger 
- IsBrowser (input -- textual representation of an image): A tool which tells, whether a current open window is a browser 
- IsEmail (input -- textual representation of an image): A tool which tells, whether a current open window is an email client 
- Verify Scam Indicators (input -- textual representation of an image): A tool which verifies, whether the text contains one or more indicators of scam form the LIST OF POSSIBLE SCAM INDICATORS:


Use the following structured reasoning format:

Question: Does this screenshot indicate any scam or security threat?

Thought: [Reason about the image]

Action: Visual Analyzer

Action Input: Current screenshot

Observation: [What Visual Analyzer detects visually]

(Repeat Thought / Action / Observation up to 5 times if needed)

Thought: I now know the final answer.

Action: Final Answer.

Scam Probability Scale:
- 0 - No scam signs: Trusted, verified source; no signs of scam.
- 1 - Uncertain: No immediate red flags, but not enough data to confirm safety.
- 2 - Low scam probability: Weak anomalies; likely safe, but caution advised.
- 3 - Moderate scam probability: Suggests possible scam/fraud. Review is needed. ANY popups about prizes, rewards, or milestones MUST be at least this level.
- 4 - High scam probability: Strong indicators of malicious or scam-related activity.
- 5 - Critical scam probability: Confirmed scam, malware, or phishing attempt. Urgent action required.


Final Answer Format:
{  
"reasoning": "<...>",
"scam_indicators": "[indicators]"
"scam_type": "<type of scam, if applicable>",
"Scam Probability": <integer from 0 to 5>
}  
"""

logger = EvaluationLogger()

def encode_image_to_data_uri(path):
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"

def full_test():
    print("SUPPOSED TO BE LEGIT:")
    for i in range(0, 451):
        if i < 10:
            i = "0" + str(i)
        image_path= "data/legit2/00" + str(i) + ".png"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        model = "llama-4-maverick-17b-128e-instruct-fp8"
        image_data_uri = encode_image_to_data_uri(image_path)
        message = {
            "role": "user",
            "content": [
                { "type": "text", "text":  TEST3 },
                { "type": "image_url", "image_url": { "url": image_data_uri } },
            ],
        }
        chat_response = client.chat.completions.create(
            model=model,
            messages=[message],
        )
        print("Chat completion output:", chat_response.choices[0].message.content)
        logger.log({
            "image":         image_path,
            "extracted_text": "",
            "response":      chat_response.choices[0].message.content
        })


    print("SUPPOSED TO BE SCAM:")
    for i in range(0, 451): 
        if i < 10:
            i = "0" + str(i)
        image_path= "data/scam2/00" + str(i) + ".png"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        image_data_uri = encode_image_to_data_uri(image_path)
        message = {
            "role": "user",
            "content": [
                { "type": "text", "text":  TEST3 },
                { "type": "image_url", "image_url": { "url": image_data_uri } },
            ],
        }
        chat_response = client.chat.completions.create(
            model=model,
            messages=[message],
        )
        print("Chat completion output:", chat_response.choices[0].message.content)
        
        logger.log({
            "image":         image_path,
            "extracted_text": "",
            "response":      chat_response.choices[0].message.content
        })



full_test()