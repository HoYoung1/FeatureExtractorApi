# ReidentifierMars128Api

## Introduce

I've extracted only the feature extraction part from the [deep sort](https://github.com/nwojke/deep_sort) project.

run.py returns 128 feature vectors from input image.

return json in run.py:
```json
{
  "feature" : [0.18031658232212067, -0.07481961697340012, -0.04913021996617317, -0.153976708650589, 0.09021788090467453, ...]
}
``` 

## Instruction

```bash
$ pip3 install -r requirements.txt
```

## How To Start

```bash
$ python3 run.py # server
```

```bash
$ python3 test/test_reid_mock_api.py # client 
```



