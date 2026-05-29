from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.llm.tool_context import ToolFlag
from typing import List

class MyCustomAgent(Agent):

    LIST_OF_ORDERS = {
        8954036174 : {
            "cheese" : "qunatity: 1",
            "butter" : "qunatity: 3",
            "banana" : "qunatity: 12"
        },

        9012018844 : {
            "apple" : "qunatity: 1",
            "mango" : "qunatity: 3",
            "banana" : "qunatity: 6"
        }
    }

    def __int__(self) -> None:
        super().__init__(
            instructions= """You are a helpful tool calling agent designed to help the user.
            Workflow:
            - Understand the user request.
            - If a tool is available that an help the user, call the respoctive tool.
            - Always try to call a tool before using your own memory to answer the user query.
            - Call a tool only if you are a hundred percent sure.
            """,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the user and tell them to ask you anything. You can specifically help them fetch the list of their orders or remove any item from their order list."
        )

    # @function_tool(flags= ToolFlag.IGNORE_ON_ENTER)
    # async def show_salary(self, name: str, ctx: RunContext) -> str:
    #     """
    #     This tool returns the salary of any person.

    #     Args:
    #         name(str) -> Name of the Person.
    #         ctx(RunContext) -> Context of the conversation

    #     Returns:
    #         string
    #     """

    #     ctx.disallow_interruptions()

    #     salary_db = {
    #         "Jack": 25000,
    #         "John": 310104
    #     }

    #     salary = salary_db.get(name, None)

    #     if salary is None:
    #         return f"I dont have the information about the salary of user {name}."
        
    #     return f"The salary of {name} is ${salary}."


    @function_tool(flags= ToolFlag.IGNORE_ON_ENTER)
    async def fetch_order_list(self, phone_number: int, ctx: RunContext) -> str:
        """
        This tool returns the list of orders of a user using phone number.

        Args:
            phone_number(str) -> Phone number of the user.
            ctx(RunContext) -> Context of the conversation

        Returns:
            string
        """

        if phone_number is None or len(str(phone_number)) != 10:
            return "Please provide a valid phone number."
        

        order_list = self.LIST_OF_ORDERS.get(phone_number, None)


        if order_list is None:
            return f"Unable to find the orders for the provided phone number {phone_number}. Please recheck the phone number and try again."
        

        return f"The list of orders linked to the phone number {phone_number} is {order_list}"
    

    @function_tool(flags= ToolFlag.IGNORE_ON_ENTER)
    async def remove_item_from_order_list(self, phone_number: int, list_of_items: List[str], ctx: RunContext) -> str:
        """
        This tool removes items from the list of orders of a user using phone number.

        Args:
            phone_number(str) -> Phone number of the user.
            list_of_items(list) -> List of items to remove
            ctx(RunContext) -> Context of the conversation

        Returns:
            string
        """

        if phone_number is None or len(str(phone_number)) != 10:
            return "Please provide a valid phone number."
        
        if list_of_items is None or len(list_of_items) == 0:
            return "Please provide list of items to be removed from the order list."

        order_list = self.LIST_OF_ORDERS.get(phone_number, None)

        if order_list is None:
            return f"Unable to find the orders for the provided phone number {phone_number}. Please recheck the phone number and try again."

        try:
            for item in list_of_items:
                order_list.pop(item)

            self.LIST_OF_ORDERS.update(phone_number, order_list)

            return f"Successfully removed the items from the order list, the updated list for phone number {phone_number} is {order_list}"

        except Exception as e:
            logger.exception(str(e))
            return "An Unexpected error occured while removing items from the order list. Please try again later."
        

