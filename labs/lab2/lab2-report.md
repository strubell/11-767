# Lab 2: Project Workshopping / Peer Feedback
---

*NoName*
Patrick Fernandes, Jared Fernandez, Haoming Zhang, Hao Zhu
---


Group name:
---
Group members present in lab today:

1: Review 1
----
Name of team being reviewed: Hagrids
1. How does your team's background align with that of the proposed project (e.g. if the proposed project is in CV but your team's background is mostly NLP, state that here. Similarly, if your team also has someone who specializes in the area of the proposed project, state that here.)
    - This project is about speech recognition, which doesn’t align well with our background. 
2. Are the motivation, hypotheses, and plan to test those hypotheses clear? If not, where could the team provide more clarification or explanation?
    - While the motivation makes some sense, it’s unclear the benefit of having such a system be on-device. Maybe specifying some down-stream, real-time application of phoneme recognition could make the point stronger
3. Does the project seem properly scoped for a semester-long project? Why or why not? How might the team be able to adjust the project to be appropriately scoped?
    - Yes. Their idea of distilling Allosaurus is very straightforward, and the number of languages is large, which makes it suitable for a semester-long project. The speaker ID system seems to be a stretch though. 
4. Are there any potential ethical concerns that arise from the proposed project?
    - No.
5. Any additional comments or concerns? Any related work you know of that might be relevant and/or helpful
    - The original Allosaurus was a very strong, zero-shot phoneme recognizer for low-resource languages. Are you planning on exploring the universality of the pruned/distilled models? I feel like this is crucial for the motivation of this project
    - What would be the benefit of actually applying the on-device phoneme recognition vs just storing the raw speech data and performing analysis later?
    - How does speaker ID make the model more useful in practice? 


2: Review 2
----
Name of team being reviewed: Masters of Science
1. How does your team's background align with that of the proposed project
    - In general there is some alignment since some group members have experience with multi-modality and most have experience with natural language processing.
2.Are the motivation, hypotheses, and plan to test those hypotheses clear? If not, where could the team provide more clarification or explanation?
    - Somewhat. The overall motivation makes sense, however it is written in a high-level, abstract way. In particular, hypotheses should be re-written as concrete, answerable questions that the proposed experiments will answer
    - Also there is a lack of specification of what particular (multi-)models they want to experiment with.
    - There is also a mention of federated learning that doesn’t translate to any actual hypothesis or experiment in the proposal. 
3. Does the project seem properly scoped for a semester-long project? Why or why not? How might the team be able to adjust the project to be appropriately scoped?
    - Iterating and training models from scratch (even without being on-device) can consume a large amount of wallclock time. Consider using a pretrained base model.
4. Are there any potential ethical concerns that arise from the proposed project?
    - The team should be careful about privacy issues when collecting human activity data
    - The team should discuss the ethical implication of the application
5. Any additional comments or concerns? Any related work you know of that might be relevant and/or helpful?
    - What’s the latency of the current cloud solution? Why are those not acceptable
    - What kind of methods will be used for optimizing models for edge computing?


3: Review 3
----
Name of team being reviewed: Multimodal Learning
1. How does your team's background align with that of the proposed project?
    - In general there is some alignment since some group members have experience with multi-modality and most have experience with natural language processing.
2. Are the motivation, hypotheses, and plan to test those hypotheses clear? If not, where could the team provide more clarification or explanation?
    - While the task is specified, there is very little detail on the dataset and model.
    - Also there is very little detail on what particular techniques they want to experiment with for optimizing the model for an on-device setting.
    - The first two hypotheses have been answered in previous literature. It would not be surprising to see positive answers to them. The team may want to further specify the hypotheses for their application
3. Does the project seem properly scoped for a semester-long project? Why or why not? How might the team be able to adjust the project to be appropriately scoped?
    - Current proposal seems limited for the scope of the semester as it primarily focuses on initial deployment of models to the device.
    - Including more details on what techniques they are going to experiment with to optimize the model could give a better impression of the scope of their proposal

4. Are there any potential ethical concerns that arise from the proposed project?
    - Project appears to be building for disabled users, the group should make sure to understand if their selected datasets are useful or provide a good case study for working with the target group.

5. Any additional comments or concerns? Any related work you know of that might be relevant and/or helpful?
    - Explicitly specify device and pretrained model. Pretrained model selection will depend on exact modalities.
    - Need a backup plan if the baseline model can't fit into the device.
    - I would also include (maybe as a stretch goal) real-time inference by outputting the result to a display or speaker. Currently no information about this is given.



4: Receiving feedback
----
Read the feedback that the other groups gave on your proposal, and discuss as a group how you will integrate that into your proposal. List 3-5 useful pieces of feedback from your reviews that you will integrate into your proposal:

1. Add multi-modality as stretch goal rather part of the core proposal
2. Be more explicit about how we are going to evaluate the proposed method on-device
3. Increased specification of initial domains and model selections.
4. Include descriptions of datasets and any necessary preprocessing for our experiments.