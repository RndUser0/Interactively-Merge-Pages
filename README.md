# Introduction by the original creator

I created a Python script that will take two doujin pages, identify all the regions where the two are different, and let you choose which parts of which page you want to merge into an output image.

>Why?

An Aqua doujin I enjoy is available in two versions: English text but mosaic censored, or Japanese text but black-bar censored. I wanted to copy all the English text onto the black-bar artwork. English text mosaic censored: "e-hentai.org/g/1603597/d0eb03920a" Japanese text black-bar censored: "Tamaya Konosuba Soushuuhen 1"

>Why not just copy & paste?

They don't perfectly line up. The English version is not an edit on the Japanese version, it was scanned separately. The pages are rotated a few degrees and some are slightly distorted in strange ways. Also, it was an interesting problem to solve.

>How do I use it?

Download the script, https://github.com/Nuthouse01/Interactively-Merge-Pages Install Python 3.7 or better: https://www.python.org/downloads/release/python-392/ Use Pip to install the necessary Python libraries (commands listed at the very top of my file). Put the file somewhere near the pages you want to merge & then double-click it. Tell it where to find the pages for base/JP version, the pages for secondary/EN version, where to put the outputs, and how many pairs of pages to merge. The "base" image should be the version that dominates the desired output; the "secondary" image should be the version that only small sections are taken from.

>How do I know this is safe?

Python code is not pre-compiled. You can read every single line of the program for yourself. Just right-click and "Open with Notepad". The only stuff you can't inspect is what's in the libraries, but Numpy, Pillow, Scikit-Image, and OpenCV are well known and widely used. If you're so paranoid that you suspect those libraries I can't say anything to comfort you.

>It doesn't work for [DOUJIN]. How do I make it work?

See that big block of allcaps variables near the top of the file? Tweak stuff until it works. If you encounter problems I didn't anticipate, you may have to do some math & programming on your own (terrifying!) If the system manages to perfectly align a large part of the image, but the edges are still misaligned, then the page was just scanned when it wasn't flat on the scanner and there's nothing we can do to perfectly fix that.

>How does it work?

Several intermediate "debug" images are posted here: https://imgur.com/a/9qhmBz4 Broadly speaking, it's a 3-step process. 1) use some very clever algorithms from OpenCV to perform feature detection, feature matching, and image alignment. 2) use a "structural similarity" metric to identify regions where the two images are significantly different, and then do some clever operations to clean & simplify those regions. 3) display an interactive window where the user can manually modify those detected regions and choose the source image for each. 4) save the resulting image & repeat for the next set of pages.

>Other notes:

AFAIK it should work with color artwork, but I've only tried it with black-and-white stuff.

**Results are not guaranteed.** This is just a cool thing I made, and I hope other people find it helpful, but nobody is paying me to make this perfect.

Expect that the parts taken from the EN/secondary image are gonna be slightly blurry, because they have been resized/distored to match the JP/base image so they're not gonna be crisp & perfect.

The difference-map-cleaning portion includes a hole-filling stage. If a textbox is right in the middle of a censored/uncensored difference, the textbox will be "swallowed up" by the difference ring around it. Look at the textbox on the dick in the top-right corner for the images I posted. In this case, you need to use the "mark as diff" or "mark as not diff" buttons to draw rectangles & manually build the difference region.

You can use the +/- keys to grow/shrink the diff region if the edges of the letters look kinda clipped. If you grow a region 10 times, then shrink it 10 times, that kinda fills in the holes of the border and makes it more rounded.

[Source](https://www.reddit.com/r/doujinshi/comments/mf5u3e/tool_for_merging_censoreduncensored_doujin_pages/)

# Credits:
* [Original creator](https://github.com/Nuthouse01/Interactively-Merge-Pages)
* [Forker](https://github.com/GymnopedieNo4/Interactively-Merge-Pages)
