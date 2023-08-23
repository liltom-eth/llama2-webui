import pytest
from llama2_wrapper.model import get_prompt_for_dialog


class TestClassGetPromptForDialog:
    from llama2_wrapper.types import Message

    dialog = []
    message1 = Message(
        role="system",
        content="You are a helpful, respectful and honest assistant. ",
    )
    message2 = Message(
        role="user",
        content="Hi do you know Pytorch?",
    )
    dialog.append(message1)
    dialog.append(message2)

    dialog2 = []
    dialog2.append(message1)
    dialog2.append(message2)
    message3 = Message(
        role="assistant",
        content="Yes I know Pytorch. ",
    )
    message4 = Message(
        role="user",
        content="Can you write a CNN in Pytorch?",
    )
    dialog2.append(message3)
    dialog2.append(message4)

    dialog3 = []
    dialog3.append(message3)
    dialog3.append(message4)
    dialog3.append(message3)
    dialog3.append(message4)
    message5 = Message(
        role="assistant",
        content="Yes I can write a CNN in Pytorch.",
    )
    dialog3.append(message5)

    def test_dialog1(self):
        prompt = get_prompt_for_dialog(self.dialog)
        # print(prompt)
        result = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. \n<</SYS>>\n\nHi do you know Pytorch? [/INST]"""
        assert prompt == result

    def test_dialog2(self):
        prompt = get_prompt_for_dialog(self.dialog2)
        # print(prompt)
        result = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. \n<</SYS>>\n\nHi do you know Pytorch? [/INST] Yes I know Pytorch. [INST] Can you write a CNN in Pytorch? [/INST]"""
        assert prompt == result

    def test_dialog3(self):
        with pytest.raises(AssertionError):
            prompt = get_prompt_for_dialog(self.dialog3)
