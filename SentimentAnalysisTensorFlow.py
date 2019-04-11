#!/usr/bin/env python
# coding: utf-8

# In[2]:


#tensorflow sentiment analysis
#lexicon,actually it the vocabulary of the words
#lets look the working of the lexicon?
#['chair','table','spoon','population']#lets assume that it is our lexicon
#i have a chair and table (now this sentense came,now what happend,we change that sentense into the vector,lets see)
#intially we assign our lexicon [0,0,0,0] 0r np.zeros(len(lexicon))#initially lexicon is zeor vector
#now what happend it read the sentense ,it check i exist in our lexicon,no,same check have and a that is in our sentense but not in lexicon
#now it check chair is exist in our lexicon,it put 1 in that same index ,[1,0,0,0]#because chair is our first index
#similarly check and it not index ,but table exist in our lexicon,so it will place 1 in that index [1,1,0,0]#bcz chair is at second index
#in this way we convert our sentence into the lexicon


# In[3]:


import nltk
from nltk.tokenize import word_tokenize
#what does the word_tokenize do?
#if we have a sentence like that (i have a chair and table) it convert into the tokens
#output will be [i,have,a,chair,and,table]
from nltk.stem import WordNetLemmatizer
#WE IS actually the WordNetLemmatizer? it acually take the exact meaning of the word,if we have run,ran,running etc,it consider it one word which is ran
#what is the stem? both stem are lemmatizer are the same,stem remove the ing,eds,etc
#the lemmatizer give the true meaning of the word,but stemming does not give us the true meaning of the word
import numpy as np
import random#bcz at somepoint we shuffle our data
import pickle#save our model
from collections import Counter
lemmatizer = WordNetLemmatizer()
hm_lines = 10000000#but here we each document 5000 lines


# In[4]:


#define a function that take our pos,neg file and apply tokenizer and lemmatizer functionality
def create_lexicon(pos,neg):
    #initially lexicon will be empty
    lexicon = []
    #read our files
    for fi in [pos,neg]:
        with open(fi,'r') as f:#open the file in read mode
            #get the lines of each file and save them into contents variables
            contents = f.readlines()
            for lines in contents[:hm_lines]:#read all the line that are exactly in our files
                all_words = word_tokenize(lines.lower())
                lexicon+=list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]#convert into the legtimate word,as we know
    word_count = Counter(lexicon)#actually it count the each word like ,happy ,8999 time aya hy and so on
    l2 =[]
    for word in word_count:
        if 1000 > word_count[word] > 50:#because for sentiment analysis we need the best word,not need 'the','a'etc
            l2.append(word)
    print(len(l2))
    return l2


def sample_handling(sample,lexicon,classification):
    feature_set = []
    with open(sample,'r') as f:
        content = f.readlines()
        for l in content[:hm_lines]:
            current_word =word_tokenize(l.lower())
            #convert that current words into the legtimate words
            current_word = [lemmatizer.lemmatize(i) for i in current_word]
            #wwe know initially the our lexicon is zeros vector [0.0.0.0],we discuss above
            features = np.zeros(len(lexicon))
            #now iterate our current_word as the sentense to match that these features exist in our lexicon or not
            #if the exist we get index from that,and place 1 their,if exist that feature in the lexicon
            for word in current_word[:hm_lines]:
                #check if the word exist in lexicon or not
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())#get the index of that word from the lexicon
                    features[index_value]+=1
                    #convert these features into the list
            features =list(features)
            feature_set.append([features,classification])
            #feature_set look like this
            #it is basicaaly the list of list
            #feature_set =[]
#             [
#              [[0,0,1,1,0,1],[0 or 1]]
#             ]
            #basically the first are our features and the other are classification,1 show the postivity and 0 for negtivity
    return feature_set

def create_feature_set_and_labels(pos,neg,test_size =0.3):
    lexicon = create_lexicon(pos,neg)
    features =[]
    features+=sample_handling('pos.txt',lexicon,[1,0])
    features+=sample_handling('neg.txt',lexicon,[0,1])
    random.shuffle(features)#shuffle is good practise for the statistical analyssi
    features =np.array(features)
    testing_size = int(test_size*len(features))#our test_size is the ten percent of the features
    train_x = list(features[:,0][:-testing_size])#get features
    train_y = list(features[:,1][:-testing_size])#get labels
    test_x = list(features[:,0][-testing_size:])#get the last ten percent
    test_y = list(features[:,1][-testing_size:])
    return train_x,train_y,test_x,test_y


# In[6]:


import tensorflow as tf
train_x,train_y,test_x,test_y =create_feature_set_and_labels('pos.txt','neg.txt')
nodes_layer1 = 500
nodes_layer2 = 500
nodes_layer3 = 500
#defien our targets
n_classes =2
batch_size =100
#define the batch_size,mean number of training example,or feature we feed to our algorithm
#lets deefine some place holders
#we may mention the placeholder size= height * width
x= tf.placeholder("float",[None,len(train_x[0])])#it is not necarry to mention its size,but just for sake if it is our input,it get the input of image in the form of 28*28 pixels
y= tf.placeholder("float")
#now start the feed forward ,we assign the weights and baise,in the key value or dictionary form
def Deep_Neural_Model(data):
    #when we defien a tensorflow variable of random normal it must contain the shape
    hidden_layer1 = {'weights':tf.Variable(tf.random_normal([len(train_x[0]),nodes_layer1])),
                    'baise':tf.Variable(tf.random_normal([nodes_layer1]))}
    hidden_layer2 = {'weights':tf.Variable(tf.random_normal([nodes_layer1,nodes_layer2])),
                    'baise':tf.Variable(tf.random_normal([nodes_layer2]))}
    hidden_layer3 = {'weights':tf.Variable(tf.random_normal([nodes_layer2,nodes_layer3])),
                    'baise':tf.Variable(tf.random_normal([nodes_layer3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([nodes_layer3,n_classes])),
                    'baise':tf.Variable(tf.random_normal([n_classes]))}
    #baise?  (input * weight ) +baise (the role of baise is if the weights and input neuron are zero,the some neuron should fire due to baise)
#   (input * weight ) +baise.. this is actually our model,that we design for each layer , let sdeign ou model
    
    l1 =tf.add(tf.matmul(data,hidden_layer1['weights']),hidden_layer1['baise'])
    #apply the activation function
    l1 = tf.nn.relu(l1)#here nn is the neural network operations
    l2 =tf.add(tf.linalg.matmul(l1,hidden_layer2['weights']), hidden_layer2['baise'])
    l2 = tf.nn.relu(l2)
    l3 =tf.add(tf.linalg.matmul(l2,hidden_layer3['weights']), hidden_layer3['baise'])
    l3 = tf.nn.relu(l3)
    output =tf.matmul(l3,output_layer['weights'] )+ output_layer['baise']
    return output#this output is the one-hot array
#at this stage we complete our computation graph

#now train our model
def train_deep_neural_model(x):#it take just the input here the x
    #make predictions
    prediction = Deep_Neural_Model(x)
    #calculate the cost or in other word loss or error .cost = target - predicted
    #tf.nn.softmax_cross_entropy_with_logits(this function is used to calculate the difference of predicted- target)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    #now our objective is that to minimize that cost,for this we use an optimizer,here we use the AdamOptimizer instead of gradient descen
    optimizer = tf.train.AdamOptimizer().minimize(cost)#this optimizer take optional parameter which is learning rate,it is by default 0.001,so we cannot modify it
    hm_epoch =20 #we know epoch =forward feed + backpropogation,actually they are cycles
    #run the seesion for computational graph
    init = tf.global_variables_initializer()
    with tf.Session() as s:
        init.run()
        for epoch in range(hm_epoch):
            epoch_lose = 0#at initially
            #now we need to find how many cycle we need on the base of total smaple and batch_size that we define above
            i=0
            while i < len(train_x):
                start =i
                end =i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _,c = s.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})#Use feed_dict To Feed Values To TensorFlow Placeholders
                epoch_lose+=c
                i+=batch_size
            print('Epoch=',epoch,'Completed out of = ',hm_epoch,'lose = ',epoch_lose)
        #to get the maximum value of prediction index and the actual label index it should be same
        correct = tf.equal(tf.arg_max(prediction,1),tf.arg_max(y,1))
        #calculate the accuracy of the model
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))#reduce mean calculte the mean,and tf.cast change the variable into the float
        print('Accuracy = ',accuracy.eval({x:test_x , y:test_y}))
    
    
    


train_deep_neural_model(x)
    

    


# In[ ]:




