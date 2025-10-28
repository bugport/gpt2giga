import asyncio

import requests
from agents import Agent, Runner, function_tool
from agents import OpenAIChatCompletionsModel
from agents import enable_verbose_stdout_logging
from openai import AsyncOpenAI

enable_verbose_stdout_logging()


client = AsyncOpenAI(base_url="http://localhost:8000", api_key="0")


@function_tool
def get_weather(city: str) -> str:
    """Получение текущей температуры через API wttr.in.
    Args:
        city (str): Название города (англ.)
    Returns:
        str: 'ясно, +20 °C' или сообщение об ошибке
    """
    url = f"https://wttr.in/{city}?format=j1"
    try:
        data = requests.get(url, timeout=60).json()
        temp_c = data["current_condition"][0]["temp_C"]
        desc = data["current_condition"][0]["weatherDesc"][0]["value"]
        return f"{desc.lower()}, {temp_c} °C"
    except Exception as e:
        return f"Ошибка: {e}"


@function_tool
def get_air_quality(city: str) -> str:
    """Возвращает PM2.5 / PM10 для заданного города.
    Args:
        city (str): Название города (англ., с большой буквы)
    Returns:
        str: 'PM2.5 = …, PM10 = …' или сообщение об ошибке
    """
    try:
        # Определяем координаты города
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en"
        geo = requests.get(geo_url, timeout=60).json()
        if not geo.get("results"):
            return "Город не найден"

        lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]

        # Качество воздуха
        aq_url = (
            "https://air-quality-api.open-meteo.com/v1/air-quality"
            f"?latitude={lat}&longitude={lon}&hourly=pm10,pm2_5&forecast_days=1"
        )
        aq = requests.get(aq_url, timeout=60).json()
        pm25, pm10 = aq["hourly"]["pm2_5"][0], aq["hourly"]["pm10"][0]
        return f"PM2.5 = {pm25} µg/m³, PM10 = {pm10} µg/m³"
    except Exception as e:
        return f"Ошибка: {e}"


spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    model=OpenAIChatCompletionsModel(model="gpt-4o", openai_client=client),
    tools=[get_weather, get_air_quality],
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model=OpenAIChatCompletionsModel(model="gpt-4o", openai_client=client),
)

russian_agent = Agent(
    name="Russian agent",
    instructions="You only speak Russian",
    model=OpenAIChatCompletionsModel(model="gpt-4o", openai_client=client),
    tools=[get_weather, get_air_quality],
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent, russian_agent],
    model=OpenAIChatCompletionsModel(model="gpt-4o", openai_client=client),
)


async def main():
    result = await Runner.run(
        triage_agent,
        input="Hola, ¿cómo es el clima en Budapest y cuál es la calidad del aire?",
    )
    print(result)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
