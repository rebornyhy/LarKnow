# LarKnow
LarKnow: a Large Language Model Improved Architecture based on Knowledge Graph Augmentation for Mechanism Interpretation of Traditional Chinese Medicine formulas

# introduction

Traditional Chinese Medicine formulas (TCMFs) play vital roles in chronic diseases due to their flexible herb compatibility and component synergy. However, the com- plexity of their numerous components and intricate interactions presents significant challenges for their mechanistic exploration. Here, we provided LarKnow, a large language model (LLM) improved architecture based on knowledge graph augmen- tation for mechanism interpretation of TCMFs. In detail, we first constructed a pharmacological knowledge graph of TCMFs (FPKG) and then generated a domain- specific question and answer dataset (TCMFQA), which was delivered to base large language models as fine-tuning materials to alleviate hallucination of these models. Notably, we proposed chain of task fine-tuning (CTF), a hybrid strategy combining chain of thought (CoT) and multi task fine-tuning (MTF). In practice, the FPKG was connected to base LLMs through the GraphRAG technology, where the CTF strat- egy was applied to guide LLMs to generate more precise and prescriptive answers for mechanistic questions of TCMFs. We validated the effectiveness of LarKnow through combining the architecture with various LLMs and comparing their per- formance with baseline models. Comparison results showed that LLMs embedded LarKnow achieved the SOTA performance in multiple tasks when compared with baseline models. Finally, the practical capability of LarKnow was evaluated through mechanism questioning and answering for TCMFs against coronary heart disease. In a nutshell, LarKnow is a novel LLM enhance architecture based on domain knowledge graph, advancing pharmacological investigation of TCMFs.

# model_pipline

![](figs\fig1.png)

# data_construction

![](figs\fig2.png)

# case_study

![](figs\fig5.png)
