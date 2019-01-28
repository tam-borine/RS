## Welcome to my blog on Remote Sensing

Here I write about stuff as I pursue my dissertation on cross-region domain adaptation of flood inundation area segmentation, using SAR images from Sentinel-1 with Google Earth Engine and TensorFlow. The goal is to build a pre-trained model capable of adapting to new unseen regions and performing comparatively better on them (dataset shift problem). 

## Posts 
- [Creating problems for yourself](#problems)
- [Autodownload shapefiles for a Copernicus EMS event](#crawler)
- [Training my first FCN to do semantic segmentation](#firstFCN)
- [Awesome Geospatial list of tools](#awesomeGeospatial)
- [Batch export to save time - GEE Python API](#batchExport)
- [The little things](#littleThings)
- [All the difference images](#diffImgs)
- [The good-bad news](#goodBadNews)
- [The good-bad news part 2](#goodBadNews2)
- [Curving in promising directions with TF eager execution](#eagerExecution)
- [New Year update](#newyearupdate)
- [Transpose Convolutions - a misnomer?](#transposeconvolutions)
- [Start here](#abstract)
- Some future post


## Start Here (or - what am I even doing?)
_28th January 2019_

I thought it was probably about time to add a little bit about what I am actually doing on a high level in case anybody actually looks at this blog and then wonders about all the low level moanings!

I understand I have given nothing in between either, no outline of my methodology or anything, more just random exclamations of weird things. But hey, you can always write to me if you want that right.

But here's an abstract! It's a draft, of course, I'm still writing things up (I love [Overleaf](https://www.overleaf.com)). So a bit busy but will share more about results soon as that's done. I'll leave you on this cliffhanger - my posts might be a bit sparser whilst I write up.

![](https://tam-borine.github.io/RS/abstract.png)

## Transpose Convolutions - a misnomer?
_22nd January 2019_

I'm implementing the final version of my CNN which is fully convolutional and inspired by the U-Net. The characteristic encoder-decoder architecture helps the problem that segmentation has (but whole-image classification does not) which is preservation of the spatial ordering of pixel values within each image, because it must be reconstructed into a segmentation map at the end. 

So basically, when you do a convolution over an input image, the output map is usually of smaller dimensions. You're capturing abstract and important features when you do this, but you are downsampling and losing resolution/ information. But because the convolution operation is discrete, if you want to reconstruct the initial image's spatial structure, which you absolutely need to for pixel-wise prediction, you can just track the ratios of your convolution matrices transformations, and do the same things on the way up.

Transpose convolution does not undo convolution operations. Deconvolution, another synonym, is even more misleading. You will not get and inverted version of whatever you started with, at any stage. This is because the kernel weights are _learned_. And whatever was learnt on the way down was forgotten. So the pixel values and weights are not having any form of inversion happening, it is only the structure that is being redistributed, literally the order of the pixels. (Although skip connections might affect this). 

So here's a great and very simple short [explanation](https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0) of Transpose Convolution, check it out!

## New Year update
_13th January 2019_

Ok so it's been ages since I've posted and feel like it's not fair to all my millions of readers (joke) considering so much has been going on!

Alright what's new.

Well something massively on my attention right now is this new ML module I've started, which is pretty awesome. I've been doing linear algebra all week, which is really fun because I didn't study math formally ever and it seems like a very useful way to think about transformations. The false Cartesian split from the OO programmer in me between functions and state has been broken forever (it was kinda wavering already from functional programming but now I'm convinced it's better), matrices as composed transformations is just so simple. Anyway I'm not going to tell you about that now other than to say 3b1b's videos are the best, and I also was confused when I learnt about dot product before linear eq systems and I have no idea why it's taught in that order. I became quite opinionated about what order it made sense to understand things in this week. We had a lecture that reviewed all of lin alg in 1h (I kid not) which I only half groked. It was a review because it covered what an undergrad math, physics or maybe cs student would do over a term at least. The lecturer acknowledged this himself and said something funny like "there are two types of people in this room, people who need to know this and people who already know this, and they are both going to hate the next hour for different reasons". Afterwards the guy sitting next to me said he did lin alg over two semesters during undergrad and that 1h covered 70% of it. Then I felt less bad about only half understanding after a week of learning.

Ok enough whining about how I don't get to do enough lin alg. Anyway theory is super helpful for my dissertation! I've been thinking more deeply about the [bias-variance trade off](https://www.dataquest.io/blog/learning-curves-machine-learning/) lately and reflecting on my learning curves (will show some screenshots next week). I've also been thinking a lot after reading stuff in the paper [A Few Useful Things to Know about Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) that should be obvious to me by now but feels like I need to hear them 1000x (check it out). And then I also found [this cool post](http://ai.stanford.edu/~zayd/why-is-machine-learning-hard.html) about why debugging in machine learning is really hard compared to regular programming (spoiler: extra dimensions of error sources are the model and the data). I leave you with his awesome diagrams (from 2d bug sources generally coding to 4d with ML). Until next week!

![](https://tam-borine.github.io/RS/regular_debugging_dimensions.png)

![](https://tam-borine.github.io/RS/plus_model_debugging_dimension.png)

![](https://tam-borine.github.io/RS/plus_data_debugging_dimension.png)

## Curving in promising directions with TF eager execution {#eagerExecution}
_25th November 2018_

If you saw my last post, you'll know that my loss curve was not looking too happy. I literally had to cheat, by giving the labels as an input feature, before my nueral net could figure anything out. So I did not actually find out what was going wrong (probably there were many things). I gave up and decided to use an approach that was more interrogatable and visible. I need to be able to debug and inspect properly. 

TF's new eager execution mode promised all of this and now seemed an apt opportunity to try it. So I did. Best decision ever. I also ditched TF's high level Estimator API in favour of Keras and writing my own simple layers and training loop. Keras is so much easier to understand what's going on and debug for me as a newb: some black boxes are too high level, others are better. I began by reading a simple [tutorial](https://medium.com/tensorflow/building-an-iris-classifier-with-eager-execution-13c00a32adb0), and could adapt it to my own data reasonably confidently because of the new transparency.

Well, it took me just a few hours to start getting accuracy that went up and loss that went down overall (albeit still very noisely - I'm working on it). Take a look!

![](https://tam-borine.github.io/RS/good_loss_curve.png)

Until next time!

## The good-bad news PART 2 {#goodBadNews2}
_15th November 2018_

So I am finally training a simple pre canned Tensorflow estimator on my data, and I literally spent the whole day yesterday staring at wild looking loss curves with no idea what was going on. I tried tweaking params such as batch size, learning rate etc. and that did definitely change the kind of wildness, but nevertheless every loss curve was definitely not going down in the slightest, and many were not better than random chance. Take a look:

![](https://tam-borine.github.io/RS/wild_training_loss.png)

There are so many things that could be going wrong here, I could not have enough data, my model could be too complex for my data, or visa versa, there could be a bug somewhere, the spatial coordinates embedding could not be accurately capturing that feature. There is a lot to work on. But honestly because I have so little experience in deep learning, I have no intuitions for what 'better' looks like. They all look wild to me. Differently, but equally, wild and random. But then, my friend gave me the idea of just cheating. Ie. give the network the answer (labels) as a feature and see if it can make use of that to cheat it's way into performing. So the good news is, I now have a lovely loss curve:

![](https://tam-borine.github.io/RS/cheating_training_loss.png)

Of course the bad news is that this model is cheating as it has the answers as an input feature, so it would be very surprising if it did not find it could just win with using that feature. But at least it narrows down (maybe, I hope) the range of possible things that could be going wrong. We know the model understands it's objective, and that it can make use of the features being inputted to it. But that's about it.

At least I have a curve to aspire to now.

## The good-bad news {#goodBadNews}
_10th November 2018_

I've been struggling with polygons forever. It really feels that way. There was the problem that came up in the first post. And many computational limit hitting from operating on vector data. Today I decided I had had enough and was going to backthumb and try out the 'convert my shapefiles to raster at the outset' approach.

In hindsight, I obviously should have tried this earlier. I thought it was going to be more complicated than it was. There are some really nice simple masking and painting functions in GEE that have just made everything so much easier. It's a little bit shocking, things are now orders of magnitude simpler and faster and I can finally be done with the preprocessing pipeline.

So the good news is I found a much simpler way that is great. The bad news is that if I'd tried this out at the beginnning (which I did actually consider briefly), I could have saved dozens of hours.

Oh well, we are where we are. Check out how simple it is:

![](https://tam-borine.github.io/RS/yay_for_rasterizing.png)

## All the difference images {#diffImgs}
_6th November 2018_

I've been upscaling the preprocessing pipeline so that it's fully automated and I can just regenerate TFRecords (the halfway checkpoint between that and my ML classification pipeline) in one go, and a bunch of stuff has occurred to me as useful that I had briefly considered but mostly glossed over in the interest of moving on. 

One of these is the question of how I choose the dates for pre-flood images given that it naturally varies from region to region with their seasons and climates. The during-flood dates are easy (although still have some variation because floods occur on different durations), but I have data on when each flood occurred so at least somewhere to go from. 

So I decided to just export a ALL THE difference images for my sites so I could visually inspect the quality of them, since for now will be the input to my classifier, it's good to get a feel for their variation. Here are some below. As you can see the quality and obviousness of flooded area is highly variable. It is unclear whether choosing pre-flood dates better (by incorporating regional climate info) is  going to help. There could be many other reasons the images that look bad have turned out that way.

We can see that Feres and Selby turned out quite nicely. I think what we see in Yangon West, Myanmar, might be an example of my poor choice of pre-flood dates. This is because we see both a bright and a dark area that looks like part of the same flood. This would naturally happen if a big flood moved, literally causing positive change in the area it came from and negative in the area it landed in. Grile shows the coastal flood but somehow the quality is really bad and it's unclear why. Also Leeds is extremely noisy and any signal is hard to see. And it is unclear why.

![](https://tam-borine.github.io/RS/Feres.png)
Feres, Greece

![](https://tam-borine.github.io/RS/YangonWest.png)
Yangon West, Myanmar

![](https://tam-borine.github.io/RS/Leeds.png)
Leeds, UK

![](https://tam-borine.github.io/RS/Grile.png)
Grile, Albania

![](https://tam-borine.github.io/RS/Selby.png)
Selby, UK


## The little things (reminders of human falibility) {#littleThings}
_3rd November 2018_

I just found a typo on Copernicus' Emergency Mapping Service website. Ok big deal. I'm not writing a post about how they should get more meticulous and how shocking it is for them to have a typo. But the way I found this out was interesting, and really just reminded me how great automation is. Not just because it's more efficient, but also because it is more repeatable and good at finding inconsistencies (like typos).

So I have some scripts that do some stuff and one just blew up trying to reconcile two lists of site names, one which was from the webpage and one taken out of the filenames themselves. It was literally one letter difference (Kyonkadun is a real place in Myanmar, Kyondadun is not...if you google them respectively you will see the only hits for the latter is [Cop EMS' webpage](http://emergency.copernicus.eu/mapping/list-of-components/EMSR130/ALL/EMSR130_15KYONDADUN) who originated the typo).

One character might not seem like a big deal, but it is. If a string doesn't match, and you have something dependening on it to do so, you flip the outcomes from True to False or vice versa. That can be a big deal. One of my favourite examples of a little bug was the [Y2K bug](https://en.wikipedia.org/wiki/Year_2000_problem), when scientists forgot about the new milleniumm and lost over 300bn dollars. 

There are definitely classes of error that are caused by the opposite, over-reliance on automation and technology, but I think on the little things humans are usually more falible. I don't even want to re-read my blog and see how many typos I have!

## Batch export to save time - GEE Python API {#batchExport}
_25th October 2018_

In my last post I was lamenting about how long some operations were taking in GEE. Saying that I needed to find other tools outside to do the job. I thought more about this and realised that not only was it the time per se that was scaring me, but the fact it was not very repeatable that concerned me most deep down. I had this little JS script doing everything: importing both vector data and S1 images, labelling, speckle filtering, doing the expensive .difference stuff, making the training set...everything. Which meant these things were not modular/small enough to A. isolate problems and B. reasonably rerun pieces of. 

So I just decided to move the really time/compute costly part out and use ee.batch (for Python API only). This turned out (after some package updating stuff) to work well (see the image below of my new labelled FeatureCollection with no overlaps). It has also encouraged me to clean up my code and move it all to Python and leave JS for mostly experiments/tinkering. I think something I've learnt, or rather been reminded of is the [KISS Principle](https://en.wikipedia.org/wiki/KISS_principle). 

![](https://tam-borine.github.io/RS/fc.png)

## Awesome Geospatial list of tools {#awesomeGeospatial}
_23rd October 2018_

Currently I'm trying to prepare/upscale the creation of my training set so that it is as fast as possible and easy to run. Whilst waiting for GEE to compute some stuff, I thought I'd take the opportunity to share [this Awesome list of geospatial tools/resources](https://github.com/sacridini/Awesome-Geospatial). Awesome lists are generally awesome and exist for many things, see the [Awesome List of Awesome Lists](https://github.com/sindresorhus/awesome) (don't worry, it stops there, otherwise it's starts getting a bit vacuous).

The reason I'm searching for new tools beyond GEE is because of that little problem I mentioned in my first post, about cutting out the flooded areas from the ROIs so we don't double label. Turns out that upscaling this is really computationally costly in GEE. It seems to be mostly caused by this particular `.difference` operation rather than the many other things I'm doing (although I am verifying that, slowly (should have started simple and then added code rather than visa versa to test contributing compute times)).

So I'm going to probably need to do that bit outside of GEE, with some geospatial Python library or something. So now I'm checking out the options and the Awesome List is helpful. In terms of my needs. I'll need something that can read in shapefiles. And something that can handle really complex multigeometries and cut them all out of the ROI efficiently. 

Will post again with updates on this. For now, enjoy the Awesome Geospatial list!

## Training my first FCN to do semantic segmentation {#firstFCN}
_11th October 2018_

It's kind of astonishing when you run something and expect it to fail in a million ways but it actually just works exactly how it should. It almost never happens, so it's actually a bit funny and confusing at first, but a welcome surprise!

I'm trying to repeat the road segmentation task using code in [this tutorial](https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef) as a way to practically try out the Fully Convolutional Network (FCN), a kind of CNN architecture that was introduced by [Long et. al](https://arxiv.org/abs/1411.4038) for the segmentation problem and uses inverse pooling to upsample because segmentation cares about spatial relationships unlike, say, plain image recognition.

I'm doing this because it's all well and good reading about the different architectural possibilities, and [there are so many great ones](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review), but until you actually get some real data in and play around with the tools, it's hard to get a feel for what actually matters and what is going to take the bulk of time to tinker with. In this case, it was just getting the data in and noticing that dependency was missing. I did it with Google Cloud Service's storage buckets, but you can just use GDrive.

If you want to see it, here's [my notebook](https://colab.research.google.com/drive/1T0h-u9sqTA21OGSsR3eRQK7_td-EtMgD) in action - it's actually training right now although unlikely anyone will see this in time, and the loss is going down most epochs!

I know you might be thinking, what does roads have anything to do with flood area segmentation? Ok, I can draw some relations but none justifiably for this research. To be honest, the idea is to just get up and running with a clean dataset that has been repeated a few times with clear baselines, before opening the world of my data. If I can get a model working well on clean data, it'll be easier to think about what is going on and have a clearer vision when swapping in my own data. I will know that the problems I encounter must be something data related, or at least data-model interaction related, as opposed to the very real possibility that I just built a crappy model for any segmentation task...even one with reasonable data. 

So the idea is to really grok my model, asking the questions and making the decisions here, and play around with the architecture with this one with the clean road segmentation data and then when ready swap my data in, and make adjustments as needed.

I'm on epoch 21/14 and my loss is at 2.386! But most of the updates were done during the backprop first epoch actually (loss from epoch 1 was 1299.525 and loss from epoch 2 was 6.633 (see screenshot), which is what you tend to see on a good learning curve I think. Exciting times!

![](https://tam-borine.github.io/RS/trainingFirstFCN.png)

Update:
There was a jump back up to 20 loss at epoch 29 but it went back down but only to ~3, which is worse than it was before whatever happened at epoch 29. This is the kind of thing that is worth investigating, and learning how to investigate now. 

## Autodownloading shapefiles for a Copernicus EMS event {#crawler}
_3rd October 2018_

For my ground truth dataset I am using vector data, shapefiles, of the inundation extent as mapped by Copernicus Emergency Mapping Service, which dedicated emergency service users can activate during all kinds of natural disaster events. There list of activations and associated maps are [here](http://emergency.copernicus.eu/mapping/list-of-activations-rapid). It is far from a complete resource, there are probably quite a few important disasters that have been missed. However, it does give us ground truth data since they use GCPs during orthorectification. **Edit** this is not sufficient to consitute ground truth data and will actually be data used for cross-referencing since that's they best we can do, just ignore wherever I say GT.

Although the website is fairly intuitive for humans, unfortunately there is no way to quickly download all their data. Ideally they would have an API or bulk download options. But instead to get the shapefiles for an event I need to click on each map's zip files, go to another page where I check a box disclaiming liability, and click another button. I figured, since I'm only going to do this once, I'll bear the brunt and do it. I needed 1.16GB worth of maps, which took me around 2 hours to manually download. 

But then, whilst filtering these locally, I accidentally deleted many of the files I needed! Because each .zip packs them up, there's no way to re-download only the subset I had accidentally deleted, so I would have to re-download everything! The idea of this was super painful. But it was a good excuse to do what I really wanted to do, an ought to have done, from the outset: automate it!

So, there were two routes:
1. see if I could just `wget` the resources (this did not work, they've done some fancy thing where there servers take POSTs not GETs for the .zip folders and also need param tokens that are generated on the fly). 
2. write some Javascript to simulate what I would be doing (you know by now this is my only option).

So guess what I did! Ok I know you just want it so here it is: 

```
var zipClassName = 'views-field-field-component-file-vectors';

var vectorfiles = document.getElementsByClassName(zipClassName);

function initiateTimeOut(i) {
    var f = vectorfiles[i];
    var url = f.lastElementChild.firstElementChild.href;
    window.newWin = window.open(url);
    setTimeout(function() { doStuff(i,window.newWin) }, 4000);
};
function doStuff(i,win) {
    console.log(win);
    win.document.getElementById('edit-confirmation').click();
    win.document.getElementById('edit-submit').click();
    i++;
    if (i <= vectorfiles.length) {
        initiateTimeOut(i); 
    }
};
  
initiateTimeOut(0);
```
If you're wondering what's with all these timeouts and functions calling each other. The answer is Javascript is weird. More [here](https://stackoverflow.com/questions/24293376/javascript-for-loop-with-timeout). 

If you want to try it out:
1. Go to some event, say this one http://emergency.copernicus.eu/mapping/list-of-components/EMSR130 
2. Right click on the page and open the browser's developer tools
3. Execute the above code (you may need to click always allow for the site as the popup blocker gets triggered in Chrome in top right hand corner - do it the first time and then repeat these steps, you won't need to do it again)

Also, there are lots of tasks this thing could generalise to. To adapt it obviously you'll need to change/remove the class names which are specific to this site (you can get these just from the page source) and also do the stuff you actually want to do on child windows in the `doStuff` function. Happy sort-of-somehow-maybe-semi-crawling!

## Creating problems for yourself {#problems}
_17th September 2018_

So I have been having some problems trying to create a training dataset in Google Earth Engine recently and thought it would be funny to share them.

I had no idea how to do this but I knew what output I wanted. Roughly something which had for every pixel in my region of interest (roi) a label of either 'flooded' or 'non-flooded', whilst keeping the image's main data (backscatter coeffcient in this case, since it's SAR). This would be stored in some appropriate data structure, which I could then export to use in Python with TF.

I found [this nice little script](https://code.earthengine.google.com/a7ed957f3034825a54b6b546b8c5ce83) from GEE's Earth Engine User Summit 2018, which I missed, which showed a straightforward way to do this.

Essentially:

- You have some Image(s)
- You have some FeatureCollection(s) which have the labels on their properties. 
- You want some structure that pairs these to create a training/test dataset for ML. Turns out `.sampleRegions()` does this.

Here's how they did it (the undefined variables were defined earlier in the script just including relevant stuff only):

```
var newfc = urban.merge(vegetation).merge(water);

var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'];

var training = image.select(bands).sampleRegions({
  collection: newfc, 
  properties: ['landcover'], 
  scale: 30
}).randomColumn();

```

So `sampleRegions` only takes one FeatureCollection (FC) which should have all the labels, for all the classes. Since I started off with two FCs, one of the whole area, which by default is unflooded, and one of the flooded area, which is a much more complicated GeometryCollection - we need to merge them. My naive approach was just to use `.merge`, but then I realised there was a problem of spatial overlap. How would any classification algorithm know which label is right for a pixel that is on an overlapped area which has a Feature of class 'flooded' and also a Feature of class 'non-flooded'? Is this representation even possible? Hopefully not as it's nonsensical! I concluded I couldn't give such a FC to `sampleRegions`. I had to somehow "cut out" the flooded FC from the non-flooded one, and then merge the resulting FC with the flooded one. I stumbled quite a bit trying to do this, thankfully [got help](https://gis.stackexchange.com/questions/295527/merge-featurecollections-excluding-overlap-in-google-earth-engine) from the GEE community on StackExchange.

The answer was to use `.difference` on the `.geometry()` of the complicated, flooded, FC. The main "problem" here was that my geometry object was super complicated, with many MultiGeometry and GeometryCollection subobjects. I kept getting server side timeouts and memory exceeded errors when trying to print the results. Eventually I thought, if GEE can't handle this, and this is only a tiny subset of the data for which I need to do this operation on, how am I ever going to get a training set? Well, as any software developer knows: when it seems tough, you just make it simpler. So I created a toy polygon called `bounds` and then limited the size of the complicated FC so that it had far fewer ee.Geometry objects. 

```
var fc = ee.FeatureCollection(table).filterBounds(bounds);

var updatedRoi = roi.map(addNonFloodedClassLabels);
var updatedFc = fc.map(addFloodedClassLabels);

var diff = ee.Feature(updatedRoi.first()).difference(updatedFc.geometry(10), 10)

var mergedFC = updatedFc.merge(diff)
```
![](https://tam-borine.github.io/RS/diff.png)

This worked! Horray! But now, how was I ever going to do it with reasonably sized data. I was tinkering around, gradually increases the size of these new bounds, when I realised that it was computing just fine, just taking a really long time! This made me think that the printing was actually the problem. Maybe the server side stuff was working and the data was processed, but just printing it here on the client was too computationally costly. In which case, maybe I don't need to worry too much just yet. I poked around and sure enough in the documentation there is [information about the unique client-server dynamic](https://developers.google.com/earth-engine/client_server) of GEE. 

On the upside, now I know that maybe computational problems are just a result of me trying to fetch massive results, but that things actually just worked sitting somewhere on the back end. On the downside, how do I get visibility into this?

I moved on, and tried to shove my mergedFC into `.sampleRegions()`. Again, I was hit with a computational error. Couldn't print the training data, exceeded 5000 elements or something. I checked the example code again. Why did they have the same amount of dimensions as their input FC? How did they keep the output dimensions so small? Turned out to be an obvious reason, they were using just a few labelled points whereas my labelled input FC, mergedFC, was many polgons covering many pixels. I discovered this by arbitrarily playing around with the `scale` param, I found when I increased it then the number of output dimensions was lower and if I made it very high, like 300, it was under 5000 - yay! However, SAR pixel spacing is 10m, not 300, and I don't want to lose information!

What to do? Well - I don't exactly know, I'm starting to think that the computational limits I am hitting are just the client side fetching ones, and that under the hood things are being processed fine. I took out [Profiler](https://developers.google.com/earth-engine/playground#profiler). Kept getting this cryptic process called `(plumbing)` at the top of my sorted usage list :/.

Looking at the Debugging guide GEE give, it seems [Export](https://developers.google.com/earth-engine/exporting) might be handy. So I'll try that next I think.

But something I learnt from all of this was that things that seem like problems are necessarily exactly the problems you think. Something could be going fine, but the real question becomes how do I get visibility into this? How much visibility is enough, what is it I need to ensure? And how do I ensure I am being as efficient as possible and not doing stupid things? 

These questions are not rhetorical, please message me! 
