# Dataset

## Annotation Format

We follow the json schema of VQA 2.0, while adding up some annotations relevant to our domain.

* "info": {"year", "version", "description", "contributor", "url", "date_created"}
* "license": {"name", "url"}
* "data_type": str(train, val or test)
* "questions": dict
    * "question_id": {"question", "video_id", "question_type"}
        * "question_type": str(a or s)
* "annotations": dict
    * "question_id": {"video_id", "answer", "ground_l", "ground_c"}
        * "ground_l": [keyword1(, keyword2)]
        * "ground_c": [coordinate1(, coordinate2)]
            * coordinate: [x1, y1, x2, y2, pov]
* "videos": dict
    * "video_id": {"begin", "end", "width", "height", "keyword"}
* "answers": list


## Data structure

All features should be stored under `/data/feat` directory.

* `/data/feat/video/(feature_type)/(video_id).pkl`
    * `data.keys()` would return [feat, cls, (four different coordinates)]
* `/data/feat/audio/(feature_type)/(video_id).pkl`
    * `data.keys()` would reaturn [feat, cls, harmonics, (time)]