import json
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse
from tqdm import tqdm
import re
import jsonlines


N_CPUS = (
    int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 1
)


#请生成一个有效的关系路径，该路径有助于回答以下问题：
gen_relpath_prompt = "你是一个中药复方知识图谱问答机器人，请生成一个或多个有效的关系路径，每个关系必须是[草药、效用、疾病、症候、症状、靶点、通路、成分、效应物]中的一个，该路径有助于回答以下问题：\n"
SOP = "<关系路径>{}</关系路径>"
SEP = "<分隔>"
ent_reg_prompt = "你是一个中药复方知识图谱问答机器人，请从输入的问题中提取有效的中药复方相关实体，该实体必须出现在输入中。\n"
SOPREG = "<实体>{}</实体>"
distur_prompt = "你是一个中药复方知识图谱问答机器人，对于输入的问题、实体、关系路径，请根据问题的语义，将关系路径分配给合适的实体，使得在知识图谱中，这些实体通过关系路径，可以搜索到问题的答案。\n"
SOPDIS = "<{}>{}</{}>"


#使用预训练的语言模型生成文本序列。函数接受模型、输入文本、分词器以及其他一些参数，然后返回生成的文本序列及其得分。
def generate_seq(
    model, input_text, tokenizer, seqs
):
    response, history = model.chat(tokenizer, input_text.strip(), max_length=256, num_beams=3, num_return_sequences=seqs, history=[])

    return {"response": response}


def token_2_rel(output):
    rel_list = []
    all_rel = ['草药','效用','疾病','症候','症状','靶点','通路','成分','效应物']
    pattern = r"<关系路径>(.*?)</关系路径>"
    for tokenstr in output:
        matched_items = re.findall(pattern, tokenstr)
        for items in matched_items:
            items_list = items.split('<分隔>')
            #print(items_list)
            isvaild = True
            for item in items_list:
                if item not in all_rel:
                    isvaild = False
            if isvaild == True:
                if items_list not in rel_list:
                    rel_list.append(items_list)
    return rel_list         

def rel_2_token(rela_list):
    rel_list = []
    for rela in rela_list:
        rel_list.append(SOP.format(SEP.join(rela)))
    return ';'.join(rel_list)


def token_2_ent(output):
    ent_list = set()
    pattern = r"<实体>(.*?)</实体>"
    matched_items = re.findall(pattern, output)
    for en in matched_items:
        ent_list.add(en)
    return list(ent_list)


def token_2_distur(output, ent_list):
    distur = dict()
    for entity in ent_list:
        distur[entity] = []
        pattern = r"<{}>(.*?)</{}>".format(entity,entity)
        matched_items = re.findall(pattern, output)
        rel_list = token_2_rel([output])
        for rel in rel_list:
            distur[entity].append(rel)
    return distur


def gen_fine_form_chatglm(data):

    gen_rel = dict()
    gen_rel['context'] = gen_relpath_prompt + data['问题']
    ent_reg = dict()
    ent_reg['context'] = ent_reg_prompt + data['问题']

    return gen_rel, ent_reg


def gen_prediction():

    tokenizer = AutoTokenizer.from_pretrained("../chatglm", trust_remote_code=True)
    model = ChatGLMForConditionalGeneration.from_pretrained("../chatglm").half().cuda()
    
    #根据参数 args.data_path、args.d、args.output_path、args.model_name 和 args.split 设置输入数据文件和输出结果的路径。
    input_file = './raw_dataset.jsonl'
    output_dir = './raw_prompts'
    print("Save results to: ", output_dir)

    # 数据
    dataset = jsonlines.open(input_file)
    dataset = list(dataset)


    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_file = os.path.join(
        output_dir, "chatglm6b.jsonl"
    )
    
    
    #处理输出文件，确保在写入新的输出之前，不会覆盖已经存在的数据。
    #f, processed_results = get_output_file(prediction_file, force=args.force)
    f = open(prediction_file,'a',encoding='utf-8')
    for data in tqdm(dataset):
        gen_rel, ent_reg = gen_fine_form_chatglm(data)
        
        pre_path_input = gen_rel["context"]
        raw_output_path = generate_seq(
            model,
            pre_path_input,
            tokenizer,
            3
        )
        pre_rel_paths = token_2_rel(raw_output_path["response"])
        
        pre_ent_input = ent_reg["context"]
        raw_output_ent = generate_seq(
            model,
            pre_ent_input,
            tokenizer,
            1
        )
        pre_ent = token_2_ent(raw_output_ent["response"][0])
        
        pre_distur_input = distur_prompt + '问题：' + data['问题'] + '\n' + '实体：' + ';'.join(pre_ent) + '\n' + '关系路径：' + rel_2_token(pre_rel_paths)
        raw_output_distur = generate_seq(
            model,
            pre_distur_input,
            tokenizer,
            1
        )
        pre_distur = token_2_distur(raw_output_distur["response"][0])
        
        if args.debug:
            print("ID: ", qid)
            print("Question: ", question)
            print("Prediction: ", rel_paths)
        # prediction = outputs[0]["generated_text"].strip()
        
        
        data['pre_path_input'] = pre_path_input
        data['raw_output_path'] = raw_output_path
        data[' pre_rel_paths'] =  pre_rel_paths
        data['pre_ent_input'] = pre_ent_input
        data['raw_output_ent'] = raw_output_ent
        data[' pre_ent'] =  pre_ent
        data['pre_distur_input'] = pre_distur_input
        data['raw_output_distur'] = raw_output_distur
        data[' pre_distur'] =  pre_distur
        
        
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()
    f.close()

    return prediction_file


if __name__ == "__main__":
    gen_path = gen_prediction()
