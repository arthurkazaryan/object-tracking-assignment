import glob
from collections import deque

import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket

from track_4 import track_data, country_balls_amount

app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
print('Started')

TRACK_DEPTH = 10


def calculate_distance(current_ball_data: np.ndarray, mean_track_data: np.ndarray):
    return np.linalg.norm(current_ball_data - mean_track_data)


def tracker_soft(el, tracking, coords):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """

    def calculate_min_idx():
        idx = None
        min_distance = int(1e+3) + 9_000  # Да да мы приколисты
        distance_list = []
        for cb_id, deque_data in coords.items():
            mean_track_data = np.array(deque_data).mean(axis=0)
            distance = np.linalg.norm(current_ball_data - mean_track_data)
            distance_list.append(distance)
            if distance < min_distance:
                min_distance = distance
                idx = cb_id

        probability = ((min_distance / distance_list) < 0.3).sum() / len(distance_list)
        return idx, probability

    for ball_data in el['data']:
        if el['frame_id'] == 1:
            ball_data['track_id'] = ball_data['cb_id']
        else:
            current_ball_data = np.array([ball_data['x'], ball_data['y']])
            data = tracking[ball_data['cb_id']]
            counts = np.bincount(data)
            track_conf = counts[np.argmax(counts)] / len(data) * 0.8

            m_idx, dist_conf = calculate_min_idx()
            if track_conf >= dist_conf:
                min_idx = np.argmax(counts)
            else:
                min_idx = m_idx
            ball_data["track_id"] = min_idx
        tracking[ball_data['cb_id']].append(ball_data["track_id"])
        coords[ball_data['cb_id']].append([ball_data['x'], ball_data['y']])

    return el, tracking


def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true для первого прогона, на повторном прогоне можете читать фреймы из папки
    и по координатам вырезать необходимые регионы.
    TODO: Ужасный костыль, на следующий поток поправить
    """
    return el


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    await websocket.send_text(str(country_balls))
    tracking = {i: deque([], TRACK_DEPTH) for i in range(country_balls_amount)}
    coords = {i: deque([], 3) for i in range(country_balls_amount)}

    for el in track_data:
        await asyncio.sleep(0.5)
        # TODO: part 1
        el, tracking = tracker_soft(el, tracking, coords)
        # TODO: part 2
        # el = tracker_strong(el)
        # отправка информации по фрейму
        await websocket.send_json(el)
    print('Bye..')
