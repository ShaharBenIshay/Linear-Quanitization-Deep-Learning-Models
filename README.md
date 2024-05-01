# Linear Quanitization for Large Deep Learning Models

**The problem:** Large deep learning models are powrful, but their size can be a burden. Storage limitations and computational costs can hinder deployment on mobile devices or resource-constrained environments.
This Large models often use high-precision floating-point numbers (such as 32-bit) for there enormous amount of parameters.
Take for example "DALL E-2" which has a 3.5 billion parameters or "Sora" which has around 3 billion parameters. 

**Naïve Solution**: "Downcasting", we can replace the high-precision floats with lower-precision ones( like 16-bit floats). This will dramatically reduce the memory usage of our model, but has a major disadvantage: can lead to an accuracy loss that will lead to lack in performance. 

**Better Solution:** Is there a way to reduce the memory usage but keep the model performance ? here comes "Linear Quantization". Quantization refers  the process of mapping a large set to a smaller set of values. 
Lets go over the steps of mapping FP32 to INT8:
1.	Map extreme values together
2.	Calculate scaling factor (S)
3.	Calculate  zero point (Z)
4.	Use S and Z to fill the rest of the the values following a linear mapping.
5.	Store params with INT8 and in inference time (test time) use dequanitization (with fp32) to avoid precision loss.  

dequanitization: If we want to go back to FP32, we can use the linear relationship that we used to quantize the original values. 

here is an image that explain what does the quanitization process is all about:

![Quantization_error](https://github.com/ShaharBenIshay/Linear-Quanitization-Deep-Learning/assets/93884611/c784ae2f-e0eb-4202-95c8-7945b3d1bef6)

Example will clarify this process: 

## So how does this "Linear Quantization" effect memory usage & performance of our model ? 

To answer those questions i will use **"Quanto"**, a python library developed by **HuggingFace** to quantize any PyTorch model. 
I will show a glimse to the answers regarding to an CLIP model and an LLM model (if you are not familiar with thos model that is fine, you can still understand the concept).

**note** - this comparison is just to have a glimse, there is a full performance comparison (with many models) done by HuggingFace

## **Model: "clip-vit-large-patch14" model by OpenAI**

* Memory Usage:
  
	original model size = 1.71 GB

	quantized model size = 0.54 GB

* Performance:

  	To comapre an CLIP model we can compare the embedding of the text & the image.
  
  	text = "a cat sitting on the beach"
  
  	image =
  
  	<img align="right" width="300" height="300" src="https://github.com/ShaharBenIshay/Linear-Quanitization-Deep-Learning/assets/93884611/8a9f284c-2470-468a-b1d4-4980b909e3fe">
	<br>
	<br>
	<br>
	<br>
	<br>
	<br>
	<br>
	<br>
	<br>
	<br>

  	Cosine Similarity for text: 0.999907
  
	Cosine Similarity for image: 0.999899

<br>
<br>

## **Model: "flan-t5-small" model by Google**

* Memory Usage:
  
	original model size = 0.307 GB

	quantized model size = 0.126 GB

* Performance:

  	To comapre an LLM model we can compare the generated words that the model has generated.

   	input text: "Hello, my name is"

   	original model output = "annie scott"
  
  	quantized model output = "annie scott"



## **Conclusion:**

Linear quantization offers a compelling solution for balancing the trade-off between model size and performance in large deep learning models.
By strategically reducing the precision of weights and activations, we can achieve significant memory savings without incurring substantial accuracy loss.

**Key takeaways:**

* Linear quantization maps high-precision floating-point values to a smaller set of integer values, resulting in a reduced model footprint.
* At inference time the model does dequanitization of the weights.
* This technique allows for deployment on resource-constrained environments like mobile devices.
* Libraries like Quanto from Hugging Face simplify the quantization process for various deep learning models.
* While a slight accuracy drop might occur, the benefits in terms of memory reduction often outweigh this drawback.













  
