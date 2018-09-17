## Welcome to my blog on Remote Sensing

Here I will write about my learnings as I explore the field of Remote Sensing (RS)/ Earth Observation (EO). Currently that means research on cross-region domain adaptation of flood inundation classification, using SAR images from Sentinel-1 with Google Earth Engine anf TensorFlow. More soon.

## Posts
- [Creating problems for yourself](#problems)
- Some future post


## Creating problems for yourself {#problems}

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

This worked! Horray! But now, how was I ever going to do it with reasonably sized data. I was tinkering around, gradually increases the size of these new bounds, when I realised that it was computing just fine, just taking a really long time! This made me think that the printing was actually the problem. Maybe the server side stuff was working and the data was processed, but just printing it here on the client was too computationally costly. In which case, maybe I don't need to worry too much just yet. I poked around and sure enough in the documentation there is [information about the unique client-server dynamic](https://developers.google.com/earth-engine/client_server) of GEE. 

On the upside, now I know that maybe computational problems are just a result of me trying to fetch massive results, but that things actually just worked sitting somewhere on the back end. On the downside, how do I get visibility into this?

I moved on, and tried to shove my mergedFC into `.sampleRegions()`. Again, I was hit with a computational error. Couldn't print the training data, exceeded 5000 elements or something. I checked the example code again. Why did they have the same amount of dimensions as their input FC? How did they keep the output dimensions so small? Turned out to be an obvious reason, they were using just a few labelled points whereas my labelled input FC, mergedFC, was many polgons covering many pixels. I discovered this by arbitrarily playing around with the `scale` param, I found when I increased it then the number of output dimensions was lower and if I made it very high, like 300, it was under 5000 - yay! However, SAR pixel spacing is 10m, not 300, and I don't want to lose information!

What to do? Well - I don't exactly know, I'm starting to think that the computational limits I am hitting are just the client side fetching ones, and that under the hood things are being processed fine. I took out [Profiler](https://developers.google.com/earth-engine/playground#profiler). Kept getting this cryptic process called `(plumbing)` at the top of my sorted usage list :/.

Looking at the Debugging guide GEE give, it seems [Export](https://developers.google.com/earth-engine/exporting) might be handy. So I'll try that next I think.

But something I learnt from all of this was that things that seem like problems are necessarily exactly the problems you think. Something could be going fine, but the real question becomes how do I get visibility into this? How much visibility is enough, what is it I need to ensure? And how do I ensure I am being as efficient as possible and not doing stupid things? 

These questions are not rhetorical, please message me! 
