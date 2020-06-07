import tkinter as tk
from tkinter import ttk, N, W, E, S
<<<<<<< HEAD
from PIL import ImageTk, Image
=======
from PIL import ImageTk, Image 
>>>>>>> ba26e9096ca33cef0daec8fefd0a95774d039ad3
import os
import random
from torchvision.transforms import CenterCrop
from torchvision.transforms.functional import resize

<<<<<<< HEAD
import ml  # plik Maksa


# os.chdir('../Wszystkie obrazki')  # katalog roboczy
# os.chdir('/home/stanislaw/datasets/open-images/train/Human body')
path = '/media/STORAGE/DATASETS/open-images/train/'
os.chdir(path)
# sciezki = [sciezka for sciezka in os.listdir('.')]  # ścieżki do obrazków

sciezki= []
for p, _, f in os.walk('.'):
    for file in f:
        sciezki.append(os.path.join(p, file))

model_name = 'pretrained'
algorithm = 1

model = ml.get_model(model_name)
transform = ml.get_transform(model_name)
if algorithm == 1:
    alg = ml.choose_dict
elif algorithm == 2:
    alg = ml.choose_dict2
else:
    alg = ml.choose_dict3


def ml_placeholder(images, model, choice, metric='cosine'):
	if type(choice) != int:
		raise Exception(f"Incorrect input: type(choice) is {type(choice)} instead of int")

	n = len(images)
	if n % 2 != 0:
		raise Exception("Incorrect input: len(obrazki) should be even number")
	n /= 2
	return random.randrange(0, n)


# Przycinanie obrazków do kwadratu (takiego jak w modelu)
def przytnij(obrazek):
	bok = 224. / 256. * min(obrazek.width, obrazek.height)
	zkwadratowany_obrazek = CenterCrop(bok)(obrazek)
	return resize(zkwadratowany_obrazek, 224)


# Wyróżnianie obrazka w pierwszym zbiorze
def wyroznij():
	global k
	k = random.randint(0, n-1)
=======
import ml # plik Maksa

def ml_placeholder(first_row, second_row, choice):
	# for i, image in enumerate(first_row):
	# 	if type(image) != Image:
	# 		raise Exception(f"Incorrect input: type(first_row[{i}]) is {type(image)} instead of PIL.Image")
	# for i, image in enumerate(second_row):
	# 	if type(image) != Image:
	# 		raise Exception(f"Incorrect input: type(second_row[{i}]) is {type(image)} instead of PIL.Image")
	if type(choice) != int:
		raise Exception(f"Incorrect input: type(choice) is {type(choice)} instead of int")
	
	return random.randrange(0, len(first_row))


main_window = tk.Tk()
main_window.title("Image guessing game")
main_window.columnconfigure(0, weight = 1)
main_window.rowconfigure(0, weight = 1)

main_frame = ttk.Frame(main_window, padding="5 5 5 5") # odstępy od krawędzi, kolejno: lewej, górnej, prawej, dolnej
main_frame.grid(column=0, row=0, sticky=(N, W, E, S))

n = 4 # TUTAJ ZMIENIAĆ
for i in range(1, n+1):
	main_frame.columnconfigure(i, weight = 1)

main_frame.rowconfigure(1, weight = 1)
main_frame.rowconfigure(3, weight = 1)


# os.chdir('../Wszystkie obrazki') # katalog roboczy
os.chdir('/home/stanislaw/datasets/open-images/train/Human body')
sciezki = [sciezka for sciezka in os.listdir('.')] # ścieżki do obrazków


#Przycinanie obrazków do kwadratu (takiego jak w modelu)
def przytnij(obrazek):
	bok = 224./256. * min(obrazek.width, obrazek.height)
	zkwadratowany_obrazek = CenterCrop(bok)(obrazek)
	return resize(zkwadratowany_obrazek, 224)

# Losowanie obrazków
sciezki_obrazkow = random.sample(sciezki, 2*n)
obrazki = [Image.open(sciezka).convert("RGB") for sciezka in sciezki_obrazkow]
photos = [ImageTk.PhotoImage(przytnij(obrazek)) for obrazek in obrazki]


# Obrazki z pierwszego zbioru
pola_obrazkow = []
for i in range(n):
	label = ttk.Label(main_frame, compound = "bottom") # albo top
	label['image'] = photos[i]
	label.grid(row=1, column=i+1)
	pola_obrazkow.append(label)

# Wyróżnianie obrazka w pierwszym zbiorze
def wyroznij():
	k = random.randint(0,n-1)
>>>>>>> ba26e9096ca33cef0daec8fefd0a95774d039ad3

	for i in range(len(pola_obrazkow)):
		pola_obrazkow[i]['borderwidth'] = 0

	pola_obrazkow[k]['borderwidth'] = 5
	pola_obrazkow[k]['relief'] = "solid"

	return k

<<<<<<< HEAD

def tworz_obrazki():
	for pole_obrazka in pola_obrazkow:
		pole_obrazka.destroy()
	for przycisk in przyciski:
		przycisk.destroy()

	pola_obrazkow.clear()
	przyciski.clear()

	for i in range(1, n+1):
		main_frame.columnconfigure(i, weight=1)

	# Obrazki z pierwszego zbioru
	for i in range(n):
		label = ttk.Label(main_frame, compound="bottom")
		label.grid(row=1, column=i+1)
		pola_obrazkow.append(label)

	# Obrazki z drugiego zbioru (przyciski)
	for i in range(n):
		przycisk = ttk.Button(main_frame, command=lambda j=i: wcisk(j))
		przycisk.grid(row=3, column=i+1)
		przyciski.append(przycisk)

	napisPolecenie.grid(row=2, column=1, columnspan=n)
	przyciskNEXT.grid(row=4, column=1, columnspan=n, pady=20)
	spinbox.grid(row=5, column=1, columnspan=n, pady=0)


def losuj_obrazki():
	global n, obrazki, photos

	n = liczba_obrazkow.get()

	sciezki_obrazkow = random.sample(sciezki, 2*n)
	obrazki = [Image.open(sciezka).convert("RGB") for sciezka in sciezki_obrazkow]
	photos = [ImageTk.PhotoImage(przytnij(obrazek)) for obrazek in obrazki]

	tworz_obrazki()

=======
k = wyroznij()

# Polecenie dla gracza
ttk.Label(main_frame, \
	text = "\nChoose an image from the row below so that the computer player could guess which image from the first row is highlighted.\n") \
	.grid(row=2, column=1, columnspan=n)


model = ml.get_model()

# Obrazki z drugiego zbioru (przyciski)
def wcisk(i):
	for przycisk in przyciski:
		przycisk.state(['!pressed'])
	przyciski[i].state(['pressed'])

	computer_guess = ml.inference(images = obrazki, model = model, choice = i, metric='cosine')
	for pole_obrazka in pola_obrazkow:
		pole_obrazka["text"] = ""
	pola_obrazkow[computer_guess]["text"] = "computer guess:" + str(computer_guess)

	

przyciski = []
for i in range(n):
	przycisk = ttk.Button(main_frame, command = lambda j=i: wcisk(j))
	przycisk['image'] = photos[n+i]
	przycisk.grid(row=3, column=i+1)
	przyciski.append(przycisk)


def losuj_obrazki():
	global obrazki, photos
	sciezki_obrazkow = random.sample(sciezki, 2*n)
	obrazki = [Image.open(sciezka) for sciezka in sciezki_obrazkow]
	photos = [ImageTk.PhotoImage(przytnij(obrazek)) for obrazek in obrazki]

>>>>>>> ba26e9096ca33cef0daec8fefd0a95774d039ad3
	for i in range(n):
		pola_obrazkow[i]['image'] = photos[i]
		przyciski[i]['image'] = photos[n+i]

	for pole_obrazka in pola_obrazkow:
		pole_obrazka["text"] = ""

<<<<<<< HEAD
	for przycisk in przyciski:
		przycisk.state(['!pressed'])

	wyroznij()


# Działanie przycisków
def wcisk(i):
	for przycisk in przyciski:
		przycisk.state(['!pressed'])
	przyciski[i].state(['pressed'])

	computer_guess = ml.inference(images=obrazki, model=model, choice=i, metric='cosine', alg=alg, transform=transform)
	# computer_guess = ml_placeholder(images=obrazki, model=model, choice=i, metric='cosine')
	for pole_obrazka in pola_obrazkow:
		pole_obrazka["text"] = ""
	if computer_guess == k:
		napis = "correctly! :D"
	else:
		napis = "incorrectly:"
	pola_obrazkow[computer_guess]["text"] = "Computer guessed " + napis


main_window = tk.Tk()
main_window.title("Image guessing game")
main_window.columnconfigure(0, weight=1)
main_window.rowconfigure(0, weight=1)

main_frame = ttk.Frame(main_window, padding="5 5 5 5")  # odstępy od krawędzi, kolejno: lewej, górnej, prawej, dolnej
main_frame.grid(column=0, row=0, sticky=(N, W, E, S))

main_frame.rowconfigure(1, weight=1)
main_frame.rowconfigure(3, weight=1)

# Polecenie dla gracza
napisPolecenie = ttk.Label(main_frame, text="\nChoose an image from the row below so that the computer can guess which image from the first row is highlighted.\n")

# przycisk NEXT
przyciskNEXT = ttk.Button(main_frame, text="NEXT IMAGES", width=20, command=losuj_obrazki)
=======
	wyroznij()

# przycisk NEXT
przyciskNEXT = ttk.Button(main_frame, text="NEXT IMAGES", width=20, command = losuj_obrazki)
przyciskNEXT.grid(row=4, column=1, columnspan=n, pady=20)
>>>>>>> ba26e9096ca33cef0daec8fefd0a95774d039ad3

# Spinbox do zmiany liczby obrazków
liczba_obrazkow = tk.IntVar(value=4)
spinbox = tk.Spinbox(main_frame, from_=1, to=10, width=3, textvariable=liczba_obrazkow, state='readonly')
<<<<<<< HEAD

pola_obrazkow = []
przyciski = []

losuj_obrazki()
=======
spinbox.grid(row=5, column=1, columnspan=n, pady=0)
>>>>>>> ba26e9096ca33cef0daec8fefd0a95774d039ad3

main_window.mainloop()