16x16 is less computationally expensive than 14x14 because you have less patches?

does the ViT still have some bias? (cnns had a lot of it because humans judged how the local context was important)

what does skip layer do? (not the same as dropout) (with many many layers , back prop weaken the training signal (the numbers become very low ). so with layer skip you provide a way to skip layers (direct linear part from output to the very first layers)) ResNET example


cnns have very high induction bias for image recognition (they will only work with images). means more structure in architecture. make learning more efficient. needs less training data


what does the CLIP do?

What CLIP is

CLIP = Contrastive Language–Image Pretraining
	•	A training setup + objective
	•	Uses two models:
	1.	Image encoder (often a ViT)
	2.	Text encoder (usually a Transformer)

Both encoders:
	•	Output vectors
	•	Are trained so matching image–text pairs have similar vectors

Think of CLIP as:

“A way to teach images and text to live in the same space.”


learnable position embedding is more flexible (the model can decide the best way to embed the position of patches)