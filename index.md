## Welcome to my blog on Remote Sensing

Here I will write about my learnings as I explore the field of Remote Sensing (RS)/ Earth Observation (EO). Currently that means research on cross-region domain adaptation of flood inundation classification, using SAR images from Sentinel-1 with Google Earth Engine anf TensorFlow. More soon.

## Posts 
- [Creating problems for yourself](#problems)
- [Autodownload shapefiles for a Copernicus EMS event](#crawler)
- [Training my first FCN to do semantic segmentation](#firstFCN)
- [Awesome Geospatial list of tools](#awesomeGeospatial)
- [Batch export to save time - GEE Python API](#batchExport)
- [The little things](#littleThings)
- Some future post


## The little things (reminders of human falibility) {#littleThinigs}
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
