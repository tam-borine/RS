## Welcome to my blog on Remote Sensing

Here I will write about my learnings as I explore the field of Remote Sensing (RS)/ Earth Observation (EO). Currently that means research on cross-region domain adaptation of flood inundation classification, using SAR images from Sentinel-1 with Google Earth Engine anf TensorFlow. More soon.

## Posts
- [Creating problems for yourself](#problems)
- Some future post


## Creating problems for yourself {#problems}

So I have been having some problems when trying to create a training dataset in Google Earth Engine recently and thought it would be funny to share them.

I had no idea how to do this but I knew what output I wanted. Roughly something which had for every pixel in my region of interest (roi) a label of either 'flooded' or 'non-flooded', whilst keeping it's main data (backscatter coeffcient in this case, since it's SAR). This would be stored in some appropriate data structure, which I could then export to use in Python with TF.

I found this nice little script from GEE's Earth Summit that I missed, which showed a way to do this.

Essentially:

- You have some Image(s)
- You have some FeatureCollection(s) which have the labels on their properties. 
- You want some structure that pairs these to create a training/test dataset for ML. Turns out `.sampleRegions()` does this.

So `sampleRegions` only takes one FeatureCollection (FC) which should have all the labels, for all the classes. Since I started off with two FCs, one of the whole area, which by default is unflooded, and one of the flooded area, which is a much more complicated GeometryCollection - we need to merge them. My naive approach was just to use `.merge`, but then I realised there was a problem of spatial overlap. How would any classification algorithm know which label is right for a pixel that is on an overlapped area which has a Feature of class 'flooded' and also a Feature of class 'non-flooded'. Is this representation even possible, hopefully not as it's nonsensical. I concluded I couldn't give such a FC to `sampleRegions`. I had to somehow "cut out" the flooded FC from the non-flooded one, and then merge the resulting FC with the flooded one. I stumbled quite a bit trying to do this, thankfully got help from the community on StackExchange.

The answer was to use `.difference` on the `.geometry()` of the complicated, flooded, FC. The main "problem" here was that it was super complicated. I kept getting server side timeouts and memory exceeded errors when trying to print the results. Eventually I thought, if GEE can't handle this, which is just a tiny subset of the data for which I need to do this, how am I ever going to do this? Well, as any good software developer does when it seems tough, you just make it simpler. So I created a toy polygon called bounds and then limited the size of the complicated FC so that it had far fewer ee.Geometry objects. This worked! Horray! But now, how was I ever going to do it with reasonably sized data. I was tinkering around, gradually increases the size of these new bounds, when I realised that it was computing just fine, just taking a really long time! This made me think that the printing was actually the problem. Maybe the server side stuff was working and the data was processed, but just printing it here on the client was too computationally costly. In which case, maybe I don't need to worry too much just yet. I poked around and sure enough in the documentation there is information about the unique client-server dynamic of GEE. 

On the upside, now I know that maybe computational problems are just a result of me trying to fetch massive results, but that things actually just worked sitting somewhere on the back end. On the downside, how do I get visibility into this?

I moved on, and tried to shove some kind of mergedFC into `.sampleRegions()`. Again, I was hit with a computational error. Exceeded 5000 elements or something. I checked the example code again. Why did they have the same amount of dimensions as their input FC? How did they keep the output dimensions so small? Turned out to be an obvious reason, they were using just a few labelled points whereas my labelled input FC, mergedFC, was many polgons covering many pixels. I discovered this by arbitrarily playing around with the `scale` param, I found when I increased it then the number of output dimensions was lower and if I made it very high, like 300, it was under 5000 - yay! However, SAR pixel spacing is 10m, not 300, and I don't want to lose information!

What to do? Well - I don't exactly know, I'm starting to think that the computational limits I am hitting are just the client side fetching ones, and that under the hood things are being processed fine. I took out Profiler. Kept getting this cryptic process called`(plumbing)`.

Looking at the Debugging guide GEE give, it seems Export might be handy.

But something I learnt from all of this was that things that seem like problems are necessarily exactly the problems you think. Something could be going fine, but the real question becomes how do I get visibility into this? How much visibility is enough, what is it I need to ensure? And how do I ensure I am being as efficient as possible and not doing stupid things? 

These questions are not rhetorical, do message me! 
