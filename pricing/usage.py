
from langchain.callbacks.base import BaseCallbackHandler
from models.database import db
from datetime import datetime

class PrintCost(BaseCallbackHandler):
    def __init__(self, username: str):
        self.username = username  # Store the user ID

    def on_llm_end(self, response, **kwargs):
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                input_cost_per_million = 4  # 4 USDC per 1M input tokens
                output_cost_per_million = 16  # 16 USDC per 1M output tokens

                input_cost = (usage.get("prompt_tokens") / 1_000_000) * input_cost_per_million
                output_cost = (usage.get("completion_tokens") / 1_000_000) * output_cost_per_million

                total_cost = input_cost + output_cost
                print("username:", self.username)
                print("input cost:", input_cost)
                print("Output Cost:", output_cost)
                print("Total Cost:", total_cost)
                collection = db.get_collection("mmosh-app-usage")
                user_doc = collection.find_one({"wallet": self.username})
                if user_doc: 
                    new_value = user_doc["value"] - total_cost
                    collection.update_one(
                    {"wallet": self.username},
                    {
                        "$set": {
                            "value": new_value,
                            "updated_date": datetime.utcnow()
                        }
                    }
    )
