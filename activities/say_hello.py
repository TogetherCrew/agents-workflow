from temporalio import activity


@activity.defn
async def say_hello():
    return 7
