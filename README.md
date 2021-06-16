# Emotion_Recognition_DNN
Tensorflow(VGG-16, Densenet-121, Densenet-201, Resnet-50, Resnet-101), Pytorch(Resnet-18, Resnet-50, WideResnet-101, Densenet-201, EfficientNet-B5, SqueezeNet1.1)
Don't forget to modify the path setting since I didn't fix it.

## Tensorflow Result

### VGG-16
![VGG-16(Tensorflow)](https://user-images.githubusercontent.com/53389350/122148185-0e484b80-ce95-11eb-9e2c-ef53962bf467.png)
### Densenet-121
![Densenet121(Tensorflow](https://user-images.githubusercontent.com/53389350/122148007-df31da00-ce94-11eb-8390-f109af4dc683.jpg)
### Densenet-201
![Densenet201(Tensorflow)](https://user-images.githubusercontent.com/53389350/122148058-ee188c80-ce94-11eb-9231-88442d022ee2.jpg)
### Resnet-50
![Resnet50(Tensorflow)](https://user-images.githubusercontent.com/53389350/122148141-02f52000-ce95-11eb-8f05-d728ab1ce2d3.png)
### Resnet-101
![Resnet101(Tensorflow)](https://user-images.githubusercontent.com/53389350/122148167-07213d80-ce95-11eb-99d9-907c42da4c5c.jpg)
## Pytorch Result

### Densenet-201
![Densenet201(Pytorch)](https://user-images.githubusercontent.com/53389350/122148035-e6f17e80-ce94-11eb-8291-3e1f4e297a72.jpg)
### EfficientNet-B5
![EfficientNetB5(Pytorch)](https://user-images.githubusercontent.com/53389350/122148074-f2dd4080-ce94-11eb-90e0-67a5647626cb.jpg)
### SqueezeNet1.1
![Squeezenet1 1(Pytorch)](https://user-images.githubusercontent.com/53389350/122148175-0a1c2e00-ce95-11eb-88c4-213defb4beab.jpg)
### Resnet-18
![resnet18(pytorch)](https://user-images.githubusercontent.com/53389350/122148089-f7a1f480-ce94-11eb-9356-e5430af1ed01.jpg)
### Resnet-50
![Resnet50(Pytorch)](https://user-images.githubusercontent.com/53389350/122148105-fcff3f00-ce94-11eb-81a9-e19955df7b02.jpg)
### WideResnet-101
![WideResnet101(Pytorch)](https://user-images.githubusercontent.com/53389350/122148198-12746900-ce95-11eb-92b5-ab23e5386218.jpg)

## Conclusion
Deep learning models implemented by tensorflow have a canonical graph, but pytorch has not. I think input shape is the main reason why pytorch didn't work properly. 
I can set input shape with tensorflow, but pytorch can't. I gathered 48X48 image data and I could set tensorflow's input shape as 48X48. But I couldn't set pytorch's 
input shape so I had no choice but to train model with 256x256 size. The data I have is 48X48, but the input shape was 256X256, so I think it cannot have been learned properly.
However, VGG-16 was the best result with 89% validation acc. However, it is hard to conclude that it is the best because it is a narrow gap with densenet and resnet.

## Data Set
![데이터셋](https://user-images.githubusercontent.com/53389350/122148987-4e5bfe00-ce96-11eb-8774-a56174f3a982.png)

I used commonly used datasets for the development of sentiment analyzers. The FER-2013 Faces Database. They are approximately 29000 image data, 48x48 size and classified into seven emotions. But One emotion, disgust, has not enough images so I decided to exclude it and train AI. This is because I heard that imbalance in datasets affects accuracy.
However, We go through the pre-process of resizing and arranging data for easy learning. Furthermore, image augmentation (zoom_range = 0.2, horizontal_flip=True, shear_range=0.2) is applied so that arbitrary data can be clearly classified.

## Face Detection
![FaceRecognition](https://user-images.githubusercontent.com/53389350/122148956-44d29600-ce96-11eb-86c2-abaf66836dc1.jpg)

I used OpenCV library to recognize faces. I just added a little code to extract the face coordinates.

## Demonstration View in Python

![파이썬 시연결과](https://user-images.githubusercontent.com/53389350/122148683-e3122c00-ce95-11eb-82ce-d413c0572ed4.png)

The above photo shows the analysis result and photo printed in a Python environment in a softmax manner.

## Reference Document

[1] Seung Hyeog Moon, “Analysis of AI-Applied Industry and Development Direction”, The Journal of the Convergence on Culture Technology(JCCT), 5:1, 77-82, 2019.
[2] M. H. Jang, “Unplugged Education Program for Artificial Intelligence Education in Elementary Schools-Focus on ‘Constraint satisfaction problem’-”, Master D. thesis, Gyeongin National University of Education, Incheon, 2020.
[3] Y. Adini, Y. Moses & S. Ullman, “Face Recognition: The problem of compensating for changes in illumination direction, IEEE Trans”. on Pattern Analysis and Machine Intelligence, 19(7), 721-732, 1997.
[4] S. H. Kim, H. H. Kim & H. S. Lee. “An Improved Face Recognition Method Using SIFT-Grid”. Journal of Digital Convergence, 11(2), 299-307, 2013.
[5] Liu, B., Sentiment, “Analysis : mining opinions, senti- ments, and emotions”, Cambridge University Press, 2015.
[6] Suh, S. and Kim, J., “Research Trend of Deep Learningbased Sentiment Analysis”, Journal of Korea Multimedia Society, Vol. 20, No. 3, pp. 8-22, 2016.
[7] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K., “BERT : Pre-training of Deep Bidirectional Transformers for Language Understanding”, NAACL-HLT, 2019. Yoo, S. and Jeong, O., An Intelligent Chatbot Utilizing BERT Model and Knowledge Graph, Journal of Society for e-Business Studies, Vol. 24, No. 3, pp. 87-98, 2019
[8] TTA, “Sentiment Ontology for Social Web”, Telecommunications Technology Association Report TTAK.KO-10.0639/R1, 2013.
[9] B. Liu, “Web Data Mining,” Springer, 2007.
[10] D. Hussein, “A survey on sentiment analysis challenges,” Journal of King Saud University – Engineering Sciences, Vol.30, No.4, pp.330-338, 2018. 
[11] K.-J. Lee, J.-H. Kim, H.-W. Seo, and K.-S. Ryoo, “Feature weighting for opinion classification of comments on news articles,” Journal of the Korean Society of Marine Engineering, Vol.34, No.6, pp.871-879, 2010.
[12] P. Turney, “Thumbs up or thumbs down? Semantic orientation applied to unsupervised classification of reviews,” Proceedings of the Association for Computational Linguistics, pp.417–424, 2002.
[13] E. Gilbert and C. J. Hutto, “VADER : A parsimonious rulebased model for sentiment analysis of social media text,” Proceedings of the 8th International Conference on Weblogs and Social Media, pp.216-225, 2014. 
[14] M. Taboada and J. Brooke, “Lexicon-based methods for sentiment analysis,” Computational Linguistics, Vol.37, No.2, pp.272–274, 2011. 
[15] L. Zhang, S. Wang, and B. Liu, “Deep learning for sentiment analysis: A survey,” arXiv:1801.07883, 2018. 
[16] OpenCV, https://ko.wikipedia.org/wiki/OpenCV, 2021.
[17] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, 9(8), 1997.
[18] G. Toderici, D. Vincent, N. Johnston, S. J. Hwang, D.Minnen,J. Shor, and M. Convell, “Full resolution image compression with recurrent neural networks,” arXiv preprint arXiv:1608.05148, 2016.
[19] Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556, 2015
[20] A. Krizhevsky, I. Sutskever, and G. E. Hinton. “ImageNet classification with deep convolutional neural networks,”Communications of the ACM, vol. 60, no. 6, pp. 84-90, May 2017.
[21] ] M. D. Zeiler and R. Fergus, “Visualizing and Understanding Convolutional Networks,” in Computer Vision-ECCV 2014, Springer pp. 818-833, 2014.
[22] C. Szegedy et al.,“Going deeper with convolutions,” in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.
[23] He, K., Zhang, X., Ren, S., & Sun, J. “Deep residual learning for image recognition”, In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.
[24] Automatic Hierarchical Classification of Kelps Using Deep Residual Features, https://www.researchgate.net/figure/ResNet-50-architecture-26-shown-with-the-residual-units-the-size-of-the-filters-and_fig1_338603223, 2015
[25] G. Huang, Z. Liu, “Densely connected convolutional networks,” Proc. of the IEEE conference on computer vision and pattern recognition, Vol. 1, No. 2, pp. 3, 2017.
[26] SqueezeNet: ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE,  https://arxiv.org/pdf/1602.07360v4.pdf, 2016
[27] M. Tan, Q. V. Le. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." arXiv preprint arXiv:1905.11946, 2019.
[28] Wide Residual Networks, https://arxiv.org/abs/1605.07146, 2016
[29] 1000x Faster Data Augmentation, https://bair.berkeley.edu/blog/2019/06/07/data_aug/, 2019
