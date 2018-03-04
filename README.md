# Coach.IO

An application to provide feedback on the Player in a video by identifying time stamps in the video where the 
Player lost points and generating an alternate reality where the Player might have done better.
<br>

<h2>Technologies and Keywords:</h2> <br>

<ol>
<li> LSTM predictive and generative models. </li>
<li> Modified pose estimation algorithm. Reference from: <a href="https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation">https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation</a></li>
<li> Deep Learning </li>
<li> Tennis </li>
<li> Tensorflow, Keras, Python </li>
</ol>

<h2> Methodology </h2>
<h3> LSTM MODEL </h3>

Courtesy: <a href = "http://karpathy.github.io/2015/05/21/rnn-effectiveness/">Andrej Karpathy's blog</a> <br> <br>
![alt text](https://raw.githubusercontent.com/Anirudhkashi/Caltech_hack/master/rnn.jpeg)

<h3> Our predictive model </h3>

![alt text](https://raw.githubusercontent.com/Anirudhkashi/Caltech_hack/master/Lstm_predict.jpg)

<h3> Our generative model </h3>

The generative model is similar in architecture to the predictive model, except that the output from LSTM is the next frame
in the image and it learns to generate new frames that way.

<h2> Repo models </h2>

Try a demo: <a href="http://10.9.27.240:8000"> http://10.9.27.240:8000 </a> (Works only at Hacktech network for now)

<ol>
<li> The LSTM_MODEL contains both predictive and generative LSTM networks </li>
<li> keras_Realtime_Multi-Person_Pose_Estimation is the cloned and modified repo for pose estimation of the tennis players </li>
<li> annotation_code contains the code to annotate the position of the ball </li>
<li> model_files contains saved models </li>
<li> Other files are images and data files </li>
</ol>
