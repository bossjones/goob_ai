from __future__ import annotations

import json
import logging

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from goob_ai.llm_manager import LlmManager


logger = logging.getLogger(__name__)


class UserInputEnrichment:
    def input_classifier_schema(self):
        input_classifier_schema = [
            {
                "name": "input_classifier",
                "description": """
                Classifier for handling user input
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "classification": {
                            "type": "string",
                            "description": """Your job is to help classify incoming text to help GOOB Developer Assistant AI designed for internal employees decide how to handle it.
                            If you are absolutely 100 percent sure the incoming text is a statement or annoucement that doesn't warrant a response, respond with "Not a question". Please don't select this if you are not absolutely sure.
                            If the incoming text is directed towards some specific user other than GOOB (or Goob or goob), respond with "Not for me"
                            For everything else, respond with "Provide Help"
                            If the incoming text is directed towards ADA (or Goob or goob), but you are not sure if it is a question or not, respond with "Provide Help"
                            DO NOT EVER ANSWER THE QUESTION OR STATEMENT DIRECTLY.""",
                            "enum": ["Not a question", "Not for me", "Provide Help"],
                        }
                    },
                    "required": ["classification"],
                },
            }
        ]
        return input_classifier_schema

    def input_classifier_tool(self, user_input: str) -> dict:
        """
        Determines how to classify incoming text.
        Parameters:
        user_input (str): The text to be analyzed.
        Returns:
        dict: A dictionary with a boolean value for the key 'classification'.
        """

        if not user_input:
            raise ValueError("The report text cannot be empty.")

        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Classify the incoming text"),
                    ("human", "{text}"),
                ]
            )
            model = LlmManager().llm.bind(
                function_call={"name": "input_classifier"}, functions=self.input_classifier_schema()
            )

            runnable = {"text": RunnablePassthrough()} | prompt | model

            first_response = runnable.invoke(user_input)
            content = first_response.content
            function_call = first_response.additional_kwargs.get("function_call")

            if function_call is not None:
                content = function_call.get("arguments", content)

            content_dict = json.loads(content)
            logger.info(f"Content dict from input classifier: {content_dict}")
            return content_dict
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON response: {e}")
        except Exception as e:
            # Handle other potential exceptions
            raise ValueError(f"An error occurred during classification: {e}")
