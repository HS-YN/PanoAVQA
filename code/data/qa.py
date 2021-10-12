import json
import numpy as np
from tqdm import tqdm

from exp import ex
from misc.geometry import conversion

'''
Data structure

 - annotation (json)
    - info: info(year,version,description,contributor,url,date_created)
    - license: license(name,url)
    - data_type: str(train,val,test)
    - questions: {'question_id': {'question', 'video_id','question_type'}}
     - annotations: {'question_id': {'video_id','answer','ground_l{1,2}','ground_c{1,2}'}
    - videos: {'video_id':{'begin','end','width','height','keyword'}
'''


@ex.capture()
def get_qa(tokenizer, data_path, max_length_qa, data=None, mode='train'):
    if data is None:
        data = json.load(open(data_path[mode], 'r'))
    outputs = {}

    minval = [10000 for _ in range(5)]
    maxval = [-10000 for _ in range(5)]

    for qid in tqdm(data['questions'].keys(), desc=f"Load {mode} split"):
        question = data['questions'][qid]
        annotation = data['annotations'][qid]
        #video = data['videos'][question['video_id']]

        encoded_question = tokenizer.encode(question['question'],
                                            truncation='longest_first',
                                            max_length=max_length_qa)

        xtl, ytl, xbr, ybr, pov = annotation['ground_c'][0]
        if type(pov) != int:
            # For some reason, few questions contain invalid grounding pov info
            # As a quickfix, we enforce pov to be 18 (ER) in such cases
            pov = 18
        geo_out = {}
        for geo in ['cartesian', 'angular', 'spherical', 'quaternion']:
            grounding = conversion(xtl, ytl, xbr-xtl, ybr-ytl, pov, geo)

            if geo == "cartesian":
                g = grounding
                grounding = [g[0]/1920., 
                             g[1]/1080., 
                             g[2]/1920.,
                             g[3]/1080.]
                # Tackling discontinuity
                if grounding[0] > 1:
                    grounding[0] -= 1
                elif grounding[0] < 0:
                    grounding[0] += 1
            else:
                #debug = annotation['ground_c'][0]
                grounding = list(grounding)

            if geo == "angular":
                if grounding[0] > np.pi:
                    grounding[0] -= 2 * np.pi
                elif grounding[0] < -np.pi:
                    grounding[0] += 2 * np.pi
                #if grounding[0] > np.pi:
                #    print(debug, [f"{x:.4f}" for x in grounding])
            geo_out[geo] = [2.] + grounding

        ''' Checking representation bound
        for i, x in enumerate(grounding):
            if x < minval[i]:
                minval[i] = x
            if x > maxval[i]:
                maxval[i] = x
        ''' # timestamp is fixed at 2
        output = {
            'question_id': qid,
            'question_type': question['question_type'],
            'video_id': question['video_id'],
            'question': encoded_question,
            'answer': annotation['answer']
        }
        output.update(geo_out)
        outputs[qid] = output

    ''' Checking representation bound
    print(geometry)
    print([f"{x:.4f}" for x in minval])
    print([f"{x:.4f}" for x in maxval])
    assert False
    '''
    
    return outputs
