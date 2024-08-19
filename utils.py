import os
from inspect import isfunction

OBJECT_CLASSES = {
    "backpack": "backpack",
    "backpack_dog": "backpack", 
    "barn" : "barn",
    "bear_plushie" : "bear_plushie",
    "berry_bowl" : "bowl",
    "can" : "can",
    "candle" : "candle",
    "chair" : "chair",
    "clock" : "clock",
    "colorful_sneaker" : "sneaker",
    "duck_toy" : "duck_toy",
    "fancy_boot" : "boot",
    "flower" : "flower",
    "grey_sloth_plushie" : "grey_sloth_plushie",
    "monster_toy" : "monster_toy",
    "pink_sunglasses" : "sunglasses",
    "poop_emoji" : "poop_emoji",
    "rc_car" : "rc_car",
    "red_cartoon" : "red_cartoon",
    "robot_toy" : "robot_toy",
    "shiny_sneaker" : "sneaker",
    "table" : "table",
    "teapot" : "teapot",
    "teddybear": "teddybear",
    "tortoise_plushy" : "tortoise_plushy",
    "vase" : "vase",
    "wolf_plushie" : "wolf_plushie",
    "wooden_pot" : "wooden_pot",
}

LIVE_SUBJECT_CLASSES = {
    "cat" : "cat",
    "cat1" : "cat",
    "cat2" : "cat",
    "dog" : "dog",
    "dog1" : "dog",
    "dog2" : "dog",
    "dog3" : "dog",
    "dog5" : "dog",
    "dog6" : "dog",
    "dog7" : "dog",
    "dog8" : "dog",
}

IMAGENET_LIVE_CLASSES = {
    'n01530575' : 'bird',
    'n01531178' : 'bird',
    'n01532829' : 'bird',
    'n01534433' : 'bird',
    'n01537544' : 'bird',
    'n01558993' : 'bird',
    'n01560419' : 'bird',
    'n01580077' : 'bird',
    'n01582220' : 'bird',
    'n01592084' : 'bird',
    'n01601694' : 'bird',
    'n02088094' : 'dog',
    'n02088238' : 'dog',
    'n02088364' : 'dog',
    'n02088466' : 'dog',
    'n02088632' : 'dog',
    'n02089078' : 'dog',
    'n02089867' : 'dog',
    'n02089973' : 'dog',
    'n02090379' : 'dog',
    'n02090622' : 'dog',
    'n02090721' : 'dog',
    'n02123045' : 'cat',
    'n02123159' : 'cat',
    'n02480495' : 'orangutan',
    'n02480855' : 'gorilla',
    'n02481823' : 'chimpanzee',
    'n02483362' : 'gibbon',
    'n02483708' : 'monkey',
    'n02484975' : 'monkey',
    'n02486261' : 'monkey',
    'n02486410' : 'monkey',
    'n02487347' : 'monkey',
    'n02488291' : 'monkey',
    'n02488702' : 'monkey',
    'n02489166' : 'monkey',
    'n02490219' : 'monkey',
    'n02492035' : 'monkey',
    'n02492660' : 'monkey',
    'n02493509' : 'monkey',
    'n02493793' : 'monkey',
    'n02494079' : 'monkey',
    'n02497673' : 'monkey',
    'n02500267' : 'monkey',
}

IMAGENET_OBJECT_CLASSES = {
    'n02690373' : 'airliner',
    'n02692877' : 'airship',
    'n02701002' : 'ambulance',
    'n02708093' : 'clock',
    'n02814533' : 'car',
    'n02823428' : 'beer_bottle',
    'n02930766' : 'taxi',
    'n02951358' : 'canoe',
    'n03041632' : 'cleaver',
    'n03100240' : 'convertible',
    'n03180011' : 'desktop_computer',
    'n03196217' : 'digital_clock',
    'n03259280' : 'Dutch_oven',
    'n03344393' : 'fireboat',
    'n03345487' : 'fire_truck',
    'n03417042' : 'dustcart',
    'n03447447' : 'gondola',
    'n03452741' : 'piano',
    'n03485407' : 'mobile_phone',
    'n03594945' : 'jeep',
    'n03642806' : 'laptop',
    'n03658185' : 'paperknife',
    'n03662601' : 'lifeboat',
    'n03670208' : 'limousine',
    'n03796401' : 'moving_van',
    'n03832673' : 'laptop',
    'n03854065' : 'pipe_organ',
    'n03930630' : 'pickup_truck',
    'n03937543' : 'pill_bottle',
    'n03977966' : 'police_van',
    'n04111531' : 'rotisserie',
    'n04238763' : 'slipstick',
    'n04273569' : 'speedboat',
    'n04461696' : 'wrecker',
    'n04612504' : 'yawl',
    'n06359193' : 'website',
}

CUB_LIVE_CLASSES = {
    "001.Black_footed_Albatross" : "bird",
    "002.Laysan_Albatross" : "bird",
    "003.Sooty_Albatross" : "bird",
    "004.Groove_billed_Ani" : "bird",
    "005.Crested_Auklet" : "bird",
    "006.Least_Auklet" : "bird",
    "007.Parakeet_Auklet" : "bird",
    "008.Rhinoceros_Auklet" : "bird",
    "009.Brewer_Blackbird" : "bird",
    "010.Red_winged_Blackbird" : "bird",
    "011.Rusty_Blackbird" : "bird",
    "012.Yellow_headed_Blackbird" : "bird",
    "013.Bobolink" : "bird",
    "014.Indigo_Bunting" : "bird",
    "015.Lazuli_Bunting" : "bird",
    "016.Painted_Bunting" : "bird",
    "017.Cardinal" : "bird",
    "018.Spotted_Catbird" : "bird",
    "019.Gray_Catbird" : "bird",
    "020.Yellow_breasted_Chat" : "bird",
    "021.Eastern_Towhee" : "bird",
    "022.Chuck_will_Widow" : "bird",
    "023.Brandt_Cormorant" : "bird",
    "024.Red_faced_Cormorant" : "bird",
    "025.Pelagic_Cormorant" : "bird",
    "026.Bronzed_Cowbird" : "bird",
    "027.Shiny_Cowbird" : "bird",
    "028.Brown_Creeper" : "bird",
    "029.American_Crow" : "bird",
    "030.Fish_Crow" : "bird"
}

OXFORD_PET_CLASSES = {
    "Abyssinian" : "cat",
    "american_bulldog" : "dog",
    "american_pit_bull_terrier" : "dog",
    "basset_hound" : "dog",
    "beagle" : "dog",
    'Bengal' : "cat",
    'Birman' : "cat",
    'Bombay' : "cat",
    'boxer' : "dog",
    'British_Shorthair' : "cat",
    'chihuahua' : "dog",
    'Egyptian_Mau' : "cat",
    'english_cocker_spaniel' : "dog",
    'english_setter' : "dog",
    'german_shorthaired' : "dog",
    'great_pyrenees' : "dog",
    'havanese' : "dog",
    'japanese_chin' : "dog",
    'keeshond' : "dog",
    'leonberger' : "dog",
    'Maine_Coon' : "cat",
    'miniature_pinscher' : "dog",
    'newfoundland' : "dog",
    'Persian' : "cat",
    'pomeranian' : "dog",
    'pug' : "dog",
    'Ragdoll' : "cat",
    'Russian_Blue' : "cat",
    'saint_bernard' : "dog",
    'samoyed' : "dog",
    'scottish_terrier' : "dog",
    'shiba_inu' : "dog",
    'Siamese' : "cat",
    'Sphynx' : "cat",
    'staffordshire_bull_terrier' : "dog",
    'wheaten_terrier' : "dog",
    'yorkshire_terrier' : "dog",
}

ABLATION_CLASSES = {
    "dog1" : "dog",
    "cat" : "cat",
    "001.Black_footed_Albatross" : "bird",
    "003.Sooty_Albatross" : "bird",
    "clock" : "clock",
    "teddybear": "teddybear",
    "wooden_pot" : "wooden_pot",
    "tortoise_plushy" : "tortoise_plushy",
}

ART_CLASSES = {
    "art1" : "art",
    "art2" : "art",
    "art3" : "art"
}

IPER_CLASSES = {
    "person1": "person",
    "person2": "person",
    "person3": "person",
    "person4": "person"
}

# ===================================================================================================================================================

LIVE_SUBJECT_PROMPT_LIST = [
    'a {} in the jungle',
    'a {} in the snow',
    'a {} on the beach',
    'a {} on a cobblestone street',
    'a {} on top of pink fabric',
    'a {} on top of a wooden floor',
    'a {} with a city in the background',
    'a {} with a mountain in the background',
    'a {} with a blue house in the background',
    'a {} on top of a purple rug in a forest',
    'a {} wearing a red hat',
    'a {} wearing a santa hat',
    'a {} wearing a rainbow scarf',
    'a {} wearing a black top hat and a monocle',
    'a {} in a chef outfit',
    'a {} in a firefighter outfit',
    'a {} in a police outfit',
    'a {} wearing pink glasses',
    'a {} wearing a yellow shirt',
    'a {} in a purple wizard outfit',
    'a red {}',
    'a purple {}',
    'a shiny {}',
    'a wet {}',
    'a cube shaped {}'
    ]

OBJECT_PROMPT_LIST = [
    'a {} in the jungle',
    'a {} in the snow',
    'a {} on the beach',
    'a {} on a cobblestone street',
    'a {} on top of pink fabric',
    'a {} on top of a wooden floor',
    'a {} with a city in the background',
    'a {} with a mountain in the background',
    'a {} with a blue house in the background',
    'a {} on top of a purple rug in a forest',
    'a {} with a wheat field in the background',
    'a {} with a tree and autumn leaves in the background',
    'a {} with the Eiffel Tower in the background',
    'a {} floating on top of water',
    'a {} floating in an ocean of milk',
    'a {} on top of green grass with sunflowers around it',
    'a {} on top of a mirror',
    'a {} on top of the sidewalk in a crowded street',
    'a {} on top of a dirt road',
    'a {} on top of a white rug',
    'a red {}',
    'a purple {}',
    'a shiny {}',
    'a wet {}',
    'a cube shaped {}'
    ]

ART_PROMPT_LIST = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
    ]

# ===================================================================================================================================================

IMAGENET_TEMPLATES_TINY = [
    "a photo of a {}",
    ]
IMAGENET_STYLE_TEMPLATES_TINY = [
    "a photo in the style of {}",
    ]
IMAGENET_TEMPLATES_SMALL = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
    ]

IMAGENET_STYLE_TEMPLATES_SMALL = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
    ]

# ===================================================================================================================================================

def parse_templates_from_superclass(superclass, placeholder=None):

    if (superclass in OBJECT_CLASSES.values()) or (superclass in IMAGENET_OBJECT_CLASSES.values()):
        templates = OBJECT_PROMPT_LIST
    elif (superclass in LIVE_SUBJECT_CLASSES.values()) or \
         (superclass in IMAGENET_LIVE_CLASSES.values()) or \
         (superclass in CUB_LIVE_CLASSES.values()) or \
         (superclass in IPER_CLASSES.values()) or \
         (superclass in OXFORD_PET_CLASSES.values()):
        templates = LIVE_SUBJECT_PROMPT_LIST
    elif (superclass in ART_CLASSES.values()):
        templates = ART_PROMPT_LIST
    else:
        raise ValueError(f"Superclass '{superclass}' not found")
    
    if placeholder is None:
        return [i.format(superclass) for i in templates]
    
    return [i.format(f"{placeholder} {superclass}") for i in templates]


def parse_templates_class_name(target_name):
    class_mappings = [
        (OBJECT_CLASSES, OBJECT_PROMPT_LIST),
        (LIVE_SUBJECT_CLASSES, LIVE_SUBJECT_PROMPT_LIST),
        (IMAGENET_LIVE_CLASSES, LIVE_SUBJECT_PROMPT_LIST),
        (IMAGENET_OBJECT_CLASSES, OBJECT_PROMPT_LIST),
        (ART_CLASSES, ART_PROMPT_LIST),
        (CUB_LIVE_CLASSES, LIVE_SUBJECT_PROMPT_LIST),
        (OXFORD_PET_CLASSES, LIVE_SUBJECT_PROMPT_LIST),
        (IPER_CLASSES, LIVE_SUBJECT_PROMPT_LIST)
    ]

    for class_dict, templates in class_mappings:
        if target_name in class_dict:
            superclass = class_dict[target_name]
            return templates, superclass

    raise ValueError(f"Target path '{target_name}' not found")

def which_target_dataset(target_name, split="train"):
    dataset_paths = {
        **{key: os.path.join("custom_datasets/data", key) for key in OBJECT_CLASSES.keys()},
        **{key: os.path.join("custom_datasets/data", key) for key in LIVE_SUBJECT_CLASSES.keys()},
        **{key: os.path.join("custom_datasets/art_data", key) for key in ART_CLASSES.keys()},
        **{key: os.path.join("custom_datasets/iper_subset", key) for key in IPER_CLASSES.keys()},
        **{key: os.path.join("custom_datasets/imagenet_subsection", key, split) for key in IMAGENET_LIVE_CLASSES.keys()},
        **{key: os.path.join("custom_datasets/imagenet_subsection", key, split) for key in IMAGENET_OBJECT_CLASSES.keys()},
        **{key: os.path.join("custom_datasets/CUB_200_2011/images", key, split) for key in CUB_LIVE_CLASSES.keys()},
        **{key: os.path.join("custom_datasets/Oxford-IIIT-Pet", key, split) for key in OXFORD_PET_CLASSES.keys()},
    }

    if target_name in dataset_paths:
        return dataset_paths[target_name]
    else:
        raise ValueError("Unidentified target name.")

# ===================================================================================================================================================

def is_object_class(superclass):
    return superclass in OBJECT_CLASSES.values() or superclass in IMAGENET_OBJECT_CLASSES.values()


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def print_box(text):
    lines = text.split('\n')
    max_length = max(len(line) for line in lines)
    print("+" + "-" * (max_length + 2) + "+")
    for line in lines:
        print(f"| {line.ljust(max_length)} |")
    print("+" + "-" * (max_length + 2) + "+")