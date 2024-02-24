"""Client code for sending requests to Lilac Garden."""
import base64
import functools
import io
import json
from typing import Any, Callable, Iterator

import numpy as np
import requests

from .env import env
from .schema import Item
from .utils import DebugTimer

GARDEN_FRONT_GATE_URL = 'https://lilacai--front-gate-fastapi-app.modal.run'
GARDEN_ENCODING_SCHEME_HEADER = 'X-Lilac-EncodingScheme'


def _decode_b64_npy(b: bytes) -> np.ndarray:
  return np.load(io.BytesIO(base64.b64decode(b)))


DECODERS: dict[str, Callable[[bytes], Item]] = {'b64-npy': _decode_b64_npy, 'json': json.loads}


def _call_garden(endpoint_name: str, docs: list[Any], **kwargs: Any) -> Iterator[Item]:
  lilac_api_key = env('LILAC_API_KEY')

  with DebugTimer('Running garden endpoint %s' % endpoint_name):
    with requests.post(
      GARDEN_FRONT_GATE_URL + '/' + endpoint_name,
      data=json.dumps(docs),
      params={k: str(v) for k, v in kwargs.items()},
      headers={
        'Authorization': f'Bearer {lilac_api_key}',
        'X-Lilac-DocCount': str(len(docs)),
      },
      stream=True,
    ) as response:
      if response.status_code > 299:
        raise requests.HTTPError(response.text)
      decoder = DECODERS[response.headers[GARDEN_ENCODING_SCHEME_HEADER]]
      for line in response.iter_lines():
        yield decoder(line)


cluster = functools.partial(_call_garden, 'cluster')
