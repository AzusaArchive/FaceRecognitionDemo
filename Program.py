import asyncio

from Final.Forms.App import App


async def Main():
    await App(asyncio.get_event_loop()).RunAsync()


if __name__ == "__main__":
    asyncio.run(Main())
