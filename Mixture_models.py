#%matplotlib inline
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import six
import tensorflow as tf
import os
import pandas as pd

from edward.models import Categorical

from Models.GMM.Save import simple_save

from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma,
    MultivariateNormalDiag, Normal, ParamMixture, RandomVariable, Bernoulli)
from tensorflow.contrib.distributions import Distribution

plt.style.use('ggplot')


###########
#For saving models:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
#from tensorflow.python.util.tf_export import tf_export


"""
Notes til Bernouli: logits: An N-D `Tensor` representing the log-odds of a `1` event. Each
 |          entry in the `Tensor` parametrizes an independent Bernoulli distribution
 |          where the probability of an event is sigmoid(logits). Only one of
 |          `logits` or `probs` should be passed in.


"""

#Reading data:
subset= pd.read_csv('/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/6. Semester/Bachelorprojekt/PyCharm/Data/user_rec.csv', sep=",", header=0)
x_trainn=subset.values                                                  #Redefine x_train into an array instead of dataframe

#Standardizing:
from sklearn import preprocessing
X= preprocessing.scale(x_trainn)

"""
#Subset:
x_train= X[0:50,]                                                       #Defining training data
x_test= X[50:100,]                                                      #Defining test data
N_testt= len(x_test)                                                    #Defining the length of the test-set
"""

#"""
#Full dataset:
x_train= X[0:5000,]                                                  #Defining training data
x_test= X[5000:10000,]                                                   #Defining test data
N_testt= len(x_test)                                                    #Defining the length of the test-set
#"""
#tf.reset_default_graph()                                                #Necessary for loop


#Next:
#Kør clusters 10,50,250,500,1000,2000,5000
#Noter alle test-likelihoods, og sammenlign.
#Derefter indsnævres intervallet.

N = len(x_train)  # number of data points                               #Setting parameters - N is defined from the number of rows
K = 5  # number of components                                           #Setting parameters - number of clusters
D = x_train.shape[1]  # dimensionality of data                           #Setting parameters - dimension of data
ed.set_seed(42)



#Model:
pi = Dirichlet(tf.ones(K))                                              #Defining prior: Dirichlet tager nogle X'er som input, der definerer sandsynligheden for hver
                                                                        # af de k clustre. Den defineres her til 1, hviklet angiver samme ssh for alle clustre.
mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)                    #Defining mu: definerer normalfordelingen, som bruges til at vise hvor clustrene ligger
sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)          #Defining sigma squared: definerer covarians-matricen, bruges til at beskrive clustrene;
                                                                        # hældning + densitet
x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},       #Given a certain hidden component, it decides the conditional distribution for the observed
                 MultivariateNormalDiag,                                #veriable
                 sample_shape=N)
z = x.cat                                                               #z is now defined as the component prescribed to the observed variable x.




#Inference:                                                             #a conclusion reached on the basis of evidence and reasoning
T = 1000                                                                #number of MCMC samples
qpi = Empirical(                                                        #Emperical is just a sample of a set, which is good enough representation of the whole set.
    tf.get_variable(                                                    #Gets an existing variable with these parameters or create a new one.
    "qpi/params", [T, K],                                               #Setting shape to be Number of MCMC samples times number of components
    initializer=tf.constant_initializer(1.0 / K)))                      #Initializer that generates tensors with constant values
qmu = Empirical(tf.get_variable(                                        #Defining mu the same way
    "qmu/params", [T, K, D],
    initializer=tf.zeros_initializer()))
qsigmasq = Empirical(tf.get_variable(                                   #Defining sigma the same way
    "qsigmasq/params", [T, K, D],
    initializer=tf.ones_initializer()))
qz = Empirical(tf.get_variable(                                         #Defining z the same way
    "qz/params", [T, N],
    initializer=tf.zeros_initializer(),
    dtype=tf.int32))

"""
We now have the following two measures of (q)pi, (q)mu, (q)sigmasq and (q)z, where:
qpi is the Empirical distribution.
pi is the Dirichlet distribution. 
"""

#Running Gibbs Sampling:
inference = ed.Gibbs({pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz},      #Starter Gibbs sampling proceduren, hvor vi har posterior-fordelingen, samt vores emperiske
                     data={x: x_train})                                 # distributions-funktion hvor den bruger x_train som værende vores data.
inference.initialize()                                                   #Initialiserer

sess = ed.get_session()
tf.global_variables_initializer().run()

t_ph = tf.placeholder(tf.int32,[])                                      #Definerer en tom placeholder
running_cluster_means = tf.reduce_mean(qmu.params[:t_ph], 0)            #Her defineres en dynamisk reduktion af cluster means.

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)
  t = info_dict['t']
  if t % inference.n_print == 0:
    print("\nInferred cluster means:")
    print(sess.run(running_cluster_means, {t_ph: t - 1}))               #Her printes den opdaterede cluster mean for hver gang der printes.
  if _ == 200:
      pi200= pi.eval()
  if _ == 400:
      pi400= pi.eval()
  if _ == 600:
      pi600= pi.eval()
  if _ == 800:
      pi800= pi.eval()




"""
###########################
#Saving model:
#1. Virker:
saver = tf.train.Saver()
save_path = saver.save(sess, "/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/6. Semester/Bachelorprojekt/PyCharm/Saved_models/model2")
print("Model saved in path: %s" % save_path)

#2. Virker, gemmer kun pi:
saver = tf.train.Saver()
sess= ed.get_session()
sess.run(tf.global_variables_initializer())
save_path = saver.save(ed.get_session(), "/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/6. Semester/Bachelorprojekt/PyCharm/Saved_models/model3")
print("Model saved in path: %s" % save_path)


#3. Virker: http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'my_test_model',global_step=1000)
"""

"""
###########################
#Importing model:
import tensorflow as tf
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('my_test_model-1000/'))
"""
"""
#Importing the model
import tensorflow as tf
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))

#Virker ikke;
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph("/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/6. Semester/Bachelorprojekt/PyCharm/Saved_models/model2.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))

  print(sess.run('pi:0'))

#3. forsøg: http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'my_test_model',global_step=1000)


#Reading the model:
#3. forsøg:
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))


#1. Virker ikke:
with tf.Session() as sess:
    saver = tf.train.import_meta_graph("/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/6. Semester/Bachelorprojekt/PyCharm/Saved_models/model.ckpt.meta")
    tf.train.Saver.restore(sess, "/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/6. Semester/Bachelorprojekt/PyCharm/Saved_models/model.ckpt")


with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph("/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/6. Semester/Bachelorprojekt/PyCharm/Saved_models/model.ckpt.meta")
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))



#4.
tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph('my_test_model-1000.meta')
"""



#Criticism:
# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).
mu_sample = qmu.sample(100)                                                 #Tager en sample af qmu størrelse: N,K,D
sigmasq_sample = qsigmasq.sample(100)                                       #Tager en sample af qsigmasq, størrelse: N,K,D
x_post = Normal(loc=tf.ones([N, 1, 1, 1]) * mu_sample,                      #Ganger disse samples ind på vores normalfordelte posterior-fordeling
                scale=tf.ones([N, 1, 1, 1]) * tf.sqrt(sigmasq_sample))      #X_post har en størrelse bestående af rows, samples, clusters, dimensions
                                                                            #Printes x_post, så har den i virkeligheden clusters, dimensions, og dette har
                                                                            #den samples gange ned. Dvs. får hvert samole, får man et ekstra print.

x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, 100, K, 1])  #tf.reshape laver først x_train om til en 5000,1,1,8 tensor. Derefter laver
                                                                            #tf.tile det til en 5000,100,5,8 tensor.

#x_post = tf.cast(x_post, tf.float64)                                       #Laves om til float64
x_broadcasted = tf.cast(x_broadcasted, tf.float32)                          #Laves om til float64


# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)                                   #jf. hvordan x_post er defineret, dvs. som en funktion af antallet af samples
                                                                            # så får man log ssh'erne for dét antal samples.
                                                                            #Dvs. at vores log_liks på første linje, indeholder ssh. over 100 samples.
log_liks = tf.reduce_sum(log_liks, 3)                                       #Man lægger alle sample udtræk sammen. Forestil dig, at man har en række med 100
                                                                            # samples, og der oprindeligt var 100 rækker. Du har nu 10k. Alle de 100 forskellige
                                                                            #samples summeres nu, vha. sum(x,3)
#log_liks = tf.reduce_mean(log_liks, 1)                                     #Vi tager nu mean af hver af rækkerne.
log_liks = tf.reduce_logsumexp(log_liks, 1)                                 # som jeg forstår det, så tager man 100 samples, summerer dem alle, at tager
                                                                            #et gennemsnit af summen af alle samples
                                                                            #prøv at kør det igennem igen med en simpel tensor.
                                                                            #xx= tf.constant([[1., 2.,4.], [1., 2.,4]])

# Choose the cluster with the highest likelihood for each data point.
clusters = tf.argmax(log_liks, 1).eval()

#Training log likelihood:
x_neg_log_prob = (-tf.reduce_sum(x_post.log_prob(x_broadcasted)) /
                    tf.cast(tf.shape(x_broadcasted)[0], tf.float32))
x_neg_log_prob.eval()


#Posterior predictive distribution:
x_post = ed.copy(x, {pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz})

#Evaluate the posterior predictive (test-data prediction from model):
x_test = tf.cast(x_test, tf.float32)
ed.evaluate('log_likelihood', data={x_post: x_test})




#End session from for-loop
#ed.get_session().close()                                                   #Necessary for loop
#tf.Session.reset()


#Plot the clusters:
#plt.scatter(x_train[:, 5], x_train[:, 6], c=clusters, cmap=cm.bwr)
#plt.axis([-5, 100, -5, 100])
#plt.title("Predicted cluster assignments")
#plt.show()
