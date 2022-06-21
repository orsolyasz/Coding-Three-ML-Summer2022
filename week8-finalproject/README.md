# Machine Reflections: A Self-Portrait Series

![Five portraits of faces labelled strength leadership power femininity and beauty](https://github.com/orsolyasz/Coding-Three-ML-Summer2022/blob/main/week8-finalproject/Generated_Imagery/00_Final_Series/00_Machine_Reflections.jpg?raw=true "Machine Reflections")

_Machine Reflections_ is a series of AI-assisted self-portraits that explore gender bias. By using large-scale generative models to create portraits of people with specific qualities and traits, this piece investigates how these models were distorted by preconceptions built into the large datasets they were trained on. Based on gender-neutral prompts of varying complexity, like  "a successful lawyer," "portrait of a person," "portrait of beauty," "the best professor in the world," "the best teacher in the world," and dozens more, the project collected a large set of machine-generated images that reveal the often heavily gendered and stereotypical nature of these systems.

The final five images in this series, _Strength_, _Leadership_, _Power_, _Femininity_, and _Beauty_ are self-portraits created by looking into the machine as a mirror. Blending an artist-curated selection of machine-generated images with the artist's self-photograph, these images hope not only to highlight biases in large-scale machine learning systems, but also to serve as a reminder that the underlying datasets leading to these results are human-made collections of (often unfiltered) data from the internet, which is a collection of social, human activity in itself.


## Initial Inspiration

My thinking about this project started when I worked on an experimental opera project (odd as that may sound) earlier this spring, using [Katherine Crawson's VQGAN+CLIP Notebook](https://colab.research.google.com/github/justinjohn0306/VQGAN-CLIP/blob/main/VQGAN%2BCLIP%28Updated%29.ipynb?authuser=0#scrollTo=ZdlpRFL8UAlWE) to generate very artificial-feeling, somewhat trippy imagery. The project required me to experiment with a range of different text prompts quite a lot. In the process, I started noticing that some prompts seemed to produce more characteristically masculine imagery by default, even if the prompts themselves were very gender-neutral (a lot of them were non-human/non-figure concepts anyway). 

This experience got me thinking about different experiments I could run on this code -or even other models- to try to really push the machine and see where it starts recreating stereotypes. Beside this, a key inspiration/goal of mine from the beginning was to somehow introduce a high level of artist curation and control, even ideally creative manual contributions to the output (eg painting or re-drawing the images, or creating a montage of them), as I was interested in physicalizing the idea that these images, and the biases that might be revealed, are not the creations of some independent AI acting on its own in any way, as well as being interested in focusing the project on using AI as a creative tool to augment my own practice, rather than as a medium in itself. This latter thought process was what led me to focusing on self-portraiture. This genre seemed to allow creative control on my end, have me literally imprint myself into the final artwork, and also open interesting avenues for exploring gender bias. Would I need to turn more masculine to look like what the AI expects a certain characteristic to look like?


## Beginning Experimentation - Looking for the Right Generative Model

![AI generated abstract shapes and parts of human faces, including a beard](https://github.com/orsolyasz/Coding-Three-ML-Summer2022/blob/main/week8-finalproject/Generated_Imagery/03_Earlier_Tests/video-4-portrait_of_a_successful_and_serious_person_seed-10.png?raw=true)
_"Portrait of a Successful and Serious Person" - VQGAN, FacesHQ, CLIP_


As I was already somewhat familiar with Crawson's system, I started my experimentation there. Because I had never tried working with any of the alternative available models (I had always stuck with the default imagenet model only), my process this time started by generating some [videos](https://drive.google.com/drive/folders/1BS92OgD5Meqh0MDEx6qqG9PfsxZUjhrJ?usp=sharing) through different models like faceshq, wiki art, as well as imagenet as it had worked nicely before (the linked folder only has the more interesting experiments saved). Unfortunately, although I was immediately seeing some interesting results in terms of bias, my hope was really to try to focus on generating images of a single, ideally recognizable human-like face, which seemed to pose a big challenge to this system. My images were always way too fragmented and not detailed or specific enough for me to easily translate them onto a self-portrait in any way. Although I had hopes for faceshq, I couldn't really ever get it to work right - it mostly kept generating many small faces as texture. It really seemed like this system -regardless of the model selected- was just better suited for more detailed prompts that were maybe more landscape-, collage-, or object-focused (like the images I had created before).

![AI generated, distorted face of a masculine figure, with many features repeated](https://github.com/orsolyasz/Coding-Three-ML-Summer2022/blob/main/week8-finalproject/Generated_Imagery/03_Earlier_Tests/video_photorealistic_portrait_of_a_powerful_person_imgnet.png?raw=true)
_"Photorealistic Portrait of a Powerful Person" - VQGAN, IMGNET,CLIP_

![AI generated image, fragmented faces and suits](https://github.com/orsolyasz/Coding-Three-ML-Summer2022/blob/main/week8-finalproject/Generated_Imagery/03_Earlier_Tests/video-12-facehq-a_strong_and_successful_ceo.png?raw=true)
_"A Strong and Successful CEO" - VQGAN, FacesHQ, CLIP_

To see if it might help, I did also try to use an image of my self as an initialization or target image, which had somewhat better results but still did not exactly feel like what I was looking for.

[![AI gnenerated, a fragmented person figure standing by several microphones](https://github.com/orsolyasz/Coding-Three-ML-Summer2022/blob/main/week8-finalproject/Generated_Imagery/03_Earlier_Tests/video-9-faceshq-portrait_of_a_powerful_inspiring_person.png?raw=true)](https://drive.google.com/file/d/1GxhUPn2u46vvYPWNUWYjNeM0kZvalS-g/view?usp=sharing)
_"Portrait of a Powerful Inspiring Person" - VQGAN, FacesHQ, CLIP, based on prompt image - Click to open Animation_

At this phase, I felt like I really needed to explore some further options (outside of this notebook) that might produce better results. As I was more focused on an artistic investigation than having control over the code at this point, I tried to look for more ready-made systems, and tried a few sites including art-breeder. The results from these all had interesting aspects to them, but they were often lower-quality and still rarely very face-focused.

![AI generated, a mix between a mountain and an old man with a long beard](https://github.com/orsolyasz/Coding-Three-ML-Summer2022/blob/main/week8-finalproject/Generated_Imagery/03_Earlier_Tests/portrait_of_a_good_confident_leader.png?raw=true)  
_Portrait of a Good Confident Leader_

![AI generated, fragmented mix between a middle aged man in a suit and maybe a spider](https://github.com/orsolyasz/Coding-Three-ML-Summer2022/blob/main/week8-finalproject/Generated_Imagery/03_Earlier_Tests/portrait_of_an_amazing_genius.png?raw=true)  
_Portrait of an Amazing Genius_

This was when I stumbled into [Dall-e Mini](https://huggingface.co/spaces/dalle-mini/dalle-mini) -which I later realized must have been actually a fairly recent release- that got me really hopeful, having seen some extremely impressive images come out of the restricted-access full version. And -although not quite comparable to its bigger version- this system produced very exciting images even from my earliest experimentation! After a few rounds, I decided to spend a good amount of time on experimenting with this generator, to see how it responds to different prompts and which ones also become interesting images.

![A 3x3 grid of blurred/abstracted masculine figures in dark robes like lawyers or judges might wear](https://github.com/orsolyasz/Coding-Three-ML-Summer2022/blob/main/week8-finalproject/Generated_Imagery/01_Experimentation-Raw_Dall-e_Mini_Screenshots/01_dominantly_m/a%20successful%20lawyer.png?raw=true)

_"Successful Lawyer" - Dall-e Mini_

![A 3x3 grid of white masculine figures in bright colored shirts and suit jackets, many of them with their hand in a fist to their chin ](https://github.com/orsolyasz/Coding-Three-ML-Summer2022/blob/main/week8-finalproject/Generated_Imagery/01_Experimentation-Raw_Dall-e_Mini_Screenshots/01_dominantly_m/portrait%20of%20a%20successful%20serious%20person.png?raw=true)

_"Portrait of a Successful Serious Person" - Dall-e Mini_



## A Deep Dive into Dall-e Mini - Building a "Dataset" of Generated Imagery

For this part of the work, I wish I could say I was really organized or systematic about my approach to coming up with a variety of text prompts, but in reality it was mostly an intuitive, iterative process. My strategy in changing or coming up with new ideas came down to three major things:

- In coming up with new prompts, I searched Google as well as my own biased mind to think of gender stereotypes (eg doctors are men, lawyers are men, men are leaders, women take care of people etc), and then used the gender-neutral bits from these ideas as prompts. I started with a focus on prompts that I expected to produce more masculine* imagery, and then moved on to trying to figure out what would be more feminine*.
- When I found an output image interesting, I would repeat the same prompt in slightly different wording or word order, to create more versions/variations of the same kind of image output - so that I would have more options to curate from later on.
- Once I developed a better sense of the kinds of prompts I wanted to work with, I started experimenting with adding in words like "portrait," "realistic," or "photorealistic" to try to get even clearer faces/images. This seemed to work extremely well for my purposes, so most -if not all- final selected images were created this way.


_*Note: This project is intentionally focused on how certain concepts are strongly influenced by stereotypes of the gender binary. Images have been categorized as more masculine (m) or feminine (m), to refrain from reinforcing the notion that certain aspects of physical appearance would somehow be inherently more linked to men or women, while still acknowledging that these words (feminine and masculine) also have social meanings that might often be revealing of how our systems -including datasets- "think" of the roles of people of different gender identities in society. I tried to avoid needing to categorize every single face (especially that arguably a number of them are ambiguous, androgynous, or gender-neutral, so I focused instead on the number of more clearly recognizable masc or fem traits in each 3x3 grid._

Throughout this generation process, I only really saved image grids where at least 1-2 images had some potential (Dall-e Mini generates a 3x3 grid of 9 distinct images as a response to each prompt by default). As images were quite small/low quality, I was saving them simply as screenshots of the grid together with the text prompt in frame. I realize this is not at all a great system and I would be more organized about it for a larger project, but I was more focused on generating as much as possible to really get a good sense of what prompts I wanted to work with, and as I noted the images were really small anyway. This image quality loss compared to the VQGAN notebook was also a sacrifice I was happy to make in exchange for better content - I figured I could deal with upscaling later.


## Observations

#### GENDER TEST 1: +WOMAN

One experiment I wanted to try was intentionally adding in strongly feminine words, such as woman, to see how the results would change for an otherwise neutrally-worded prompt that had produced more masculine images.

{IMAGES HERE}


#### OVERBEARINGLY WEIGHTED WORDS

As I was exploring, I noticed that there were certain words that would have an overbearingly powerful effect on the image, pulling it toward their base meaning, even if in context these words were not the focus of the text prompt. Words like earth, natural, or future are examples that could simply not be included without overpowering the overall image (anything with "earth" in it just seemed to produce a massive globe with some minor details maybe), and often even take away a person/portrait element completely (even if person had been in the prompt, even as the very first word of the text). 
{IMAGES HERE}


#### GENDER TEST 2: -PERSON

After a while, I started feeling like perhaps the word "person" was also an overbearingly powerful word, pulling everything into masc territory. My dilemma became: if this was the case, that would be revealing in itself, but it would still stop me from seeing the effect of other words appropriately. So, I ran a test where the prompt was literally only to show a portrait of a person - and yes, it was overwhelmingly masc. Keeping this in mind, I then tried a number of my previous prompts without the word "person" in it, to try to really focus on the representations of the actual characteristics I was looking at.
{IMAGES HERE}


#### GENDER TEST 3: +GENDER

At one point, I wanted to see if the AI might have heard about "the weak gender" and the "strong gender" -- but to my surprise, both came out to be very non-masc images. This made me think -- is gender in itself understood as a non masculine word? Do men not have a gender according to this AI? And, indeed, gender is so strongly seen as the opposite of masc that "portrait of a person from a gender" comes out really fem, even though we just established that person was weighted toward masc meanings.
{IMAGES HERE}


#### ADDING IN "REALISTIC PORTRAIT"

Finally, as mentioned, I added in portrait-specific words to try to get better faces out of the system later on - here are some results of that.
{IMAGES HERE}



## Curation and Creation - Final Image Blending

After this phase of what was basically "data collection," now having about 129 x 9 images at my disposal, it was time to move to the curation and creation phase. The first step of this was to categorize all my data - by labelling each collection (grid of 9 pictures saved as one file) as either 1) dominantly masculine, 2) more masc than feminine, 3) neutral or non-human (this I mostly just kept for reference on how the model works), 4) more fem than masc, and 5) dominantly fem. 

Seeing that a large portion of my collection focused on heavily masculine and heavily feminine images, I realized that this was an area I really wanted to work with for the next phases of my project. I went through these folders a couple times again, this time looking for the images that really absolutely grabbed me -either for a stylistic reason, or for having clear and unique faces in them. I ended up narrowing down my top choices to about 10 images, and so now it was time to move on to applying some of these to my own portrait.
{IMAGES HERE}

For creating a "blend" with my face, I was at first really unsure what approach I wanted to take. I tried to look around and tried a few style transfer examples, but nothing really seemed to do what I wanted -- while I had picked some of my selections for their artistic style, I still felt like there needed to be some sort of a facial feature blend between the generated portrait results and my own face for my point to come across. I also tried seeing if going back to the VQGAN notebook might be a good approach, but the results it produced were still too fragmented even with just the initial and target images set.

Then I realized - I still had the problem of my generated images being quite small as well. And while they were interesting, and often unique in their "art style," they were still perhaps not "face-like" enough for a nice blend to occur. To solve these problems, I decided to go back to the )(StyleGAN example notebook)[LINK HERE] from class, and to set it up to allow me to upload number of my generated pictures as well as my portrait and try interpolating or style mixing between them. 

This process did pose a number of challenges as I had to test out with some trial and error which generated faces the algorithm was able to recognize as faces, but once I had my final 6 or so selections that could be recognized, the projection of these into w space worked really well. Better than I even planned or expected, I now had fully machine-rendered (fairly) realistic faces of what a powerful person might look like, or what a strong or beautiful person might look like (etc) -- which was really exciting. 

{IMAGES HERE}

As the last step, I created interpolations and style mixes between these projections and a projection of my own face, and then selected You can see videos of interpolations as well as the full style mix strip in the main (repo)[LINK HERE] under each appropriately named folder. 

{GIF OF TRANSITIONS HERE}


From these, I selected relatively mid-way blends where both my facial features and the projected generated image's features were clearly recognizable. For some portraits, style-mixing worked better, but for others I preferred the interpolation results and took out an ideal still from the interpolated video. After manually editing back some of the texture and style from the origin generated images (as these got lost when the faces turned more realistic as projections), I had my final five images, and saved them in a sequence I found meaningful

![Five portraits of faces labelled strength leadership power femininity and beauty](https://github.com/orsolyasz/Coding-Three-ML-Summer2022/blob/main/week8-finalproject/Generated_Imagery/00_Final_Series/00_Machine_Reflections.jpg?raw=true "Machine Reflections")


## Final Takeaways, Future Plans

Overall, while I consider the final series of five portraits the main product of this project, I am also really satisfied with the range of exploration I was able to undertake in the process. The saved set of generated images point to a range of stereotypes built in to these systems.  While Dall-e Mini has a note about potential bias and stereotypes on their platform, it is clear that these biases are not going to be easy to address on a system without direct intervention (eg intentionally removing gendered "weights" form non-gendered words) that was trained on unfiltered internet data - as ultimately, the system merely reflects the many ways in which a lot of society think and talk a biased way on the internet.

The one regret I have is not managing time enough to explore translating my final product into a manually created physical art piece. This aspect was something I had hoped to explore more, but I ran out of time having spent quite long on image generation and the experimentation with different text prompts. I definitely plan on recreating the series as a mixed media painting in the near future.

On a final, technical note, I wonder if and how I could find a way to create a more integrated system for the creation of these images - I did not really find a notebook to use dall-e mini but if I did, or found out how to do that, I would love to build the two steps - image generation and portrait blending - into one single notebook for a smoother workflow.



