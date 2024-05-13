import numpy as np
from PIL import Image, ImageDraw, ImageOps

#Mehdi Znata  &  Diab mohammad ibrahim  & Dinedane Abdelkader



# Matrice utilisée pour encoder le message
ENCODING_MATRIX = np.array([[1, 3], [2, 5]])

ENCODING_VALUES = {
    ".": 1,
    "@": 5,
    "#": 10,
    "$": 50,
    "%": 100,
    "^": 500,
    "&": 1000
}

SYMBOLS_VALUES = sorted(ENCODING_VALUES.items(), key=lambda x: x[1], reverse=True)


def encode_num(num):
    """
    Encode a number into a string representation using SYMBOLS_VALUES.

    :param num: The number to be encoded.
    :return: The encoded string representation of the number.
    """
    symbol_str = ""
    for symbol, value in SYMBOLS_VALUES:
        while num >= value:
            symbol_str += symbol
            num -= value
    return symbol_str


def decode_str(sym_str):
    """
    Decodes a symbol string by summing the values of each character in the string.

    :param sym_str: A string representing symbols.
    :return: An integer representing the sum of the values of each character in the symbol string.
    """
    return sum(ENCODING_VALUES[char] for char in sym_str)


def get_encoded_matrix_rep(encoded_matrix):
    """
    :param encoded_matrix: A 2-dimensional matrix representing the encoded data.
    :return: A 1-dimensional list containing all the elements of the encoded matrix.
    """
    return [item for sublist in encoded_matrix for item in sublist]

def encode_with_matrix_A(msg_to_encode):
    """
    :param msg_to_encode: A string representing the message to be encoded.
    :return: A matrix containing the encoded characters of the input message.

    This method takes a message and encodes it by converting each pair of characters into a matrix representation. The encoding process involves obtaining the ASCII values of each character
    *, performing matrix multiplication using a predefined encoding matrix, and then mapping the resulting values to encoded characters using the encode_num() function. The encoded matrix
    * is returned as the result.

    Example usage:

    encoded_msg = encode_with_matrix_A("Hello world")
    print(encoded_msg)

    Output:
    [[19, -4], [19, -4], [32, -16], [-4, 48], [-4, 15], [49, 4], [32, -16], [51, 0], [52, 51], [50, 12], [32, -16], [52, 99], [56, 31], [51, 0], [47, 27]]
    """
    encoded_matrix = []
    n = len(msg_to_encode)
    for i in range(0, n, 2):
        first_char_ascii = ord(msg_to_encode[i])
        second_char_ascii = ord(msg_to_encode[i + 1]) if i + 1 < len(msg_to_encode) else ord(" ")  # Blank space
        result = np.matmul(ENCODING_MATRIX, [first_char_ascii, second_char_ascii])
        encoded_matrix.append([encode_num(num) for num in result])
    return encoded_matrix


def decode_with_matrix_A(encoded_matrix):
    """
    ..
        decode_with_matrix_A(encoded_matrix)

    :param encoded_matrix: A matrix containing encoded values.
    :type encoded_matrix: list[list[int]]

    ::rtype:: str
    :return: The decoded message.

    The `decode_with_matrix_A` method decodes an encoded matrix using a pre-defined encoding matrix.

    The method performs the following steps:
    1. Calculates the inverse of the encoding matrix.
    2. Iterates over each row in the encoded matrix.
    3. Decodes each value in the row using the `decode_str` method.
    4. Computes the dot product of the inverse of the encoding matrix and the decoded row.
    5. Rounds each value in the resulting row to the nearest integer.
    6. Appends the rounded values to the `decoded_matrix` list.
    7. Joins all the integers in the `decoded_matrix` into a single string representation of the decoded message.

    Example usage:

    .. code-block:: python

        encoded_matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

        decoded_msg = decode_with_matrix_A(encoded_matrix)
        print(decoded_msg)
        # Output: 'abc'

    Note: The code assumes the existence of the `decode_str` and `ENCODING_MATRIX` variables.
    """
    decoded_matrix = []
    inv_matrix_A = np.linalg.inv(ENCODING_MATRIX)
    for row in encoded_matrix:
        row = [decode_str(num_str) for num_str in row]
        result = np.matmul(inv_matrix_A, row)
        decoded_matrix.append([int(round(num)) for num in result])
    decoded_msg = ''.join(chr(val) for sublist in decoded_matrix for val in sublist)
    return decoded_msg



class Shape:


    def __init__(self, x, y, width, height, color, isCircle=False):
        """
        :param x: position horizontale de la forme
        :param y: position verticale de la forme
        :param width: largeur de la forme
        :param height: hauteur de la forme
        :param color: couleur de la forme
        :param isCircle: booléen indiquant si la forme est un cercle ou non (par défaut False)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.isCircle = isCircle



class MessageDrawer:
    """
    Initializes a new instance of the MessageDrawer class.

    :param encoded_matrix: The encoded matrix representing the message.
    """
    # Cette classe représente un "outil" de dessin de message encodé

    def __init__(self, encoded_matrix):
        """
        :param encoded_matrix: la matrice encodée représentant le message
        """
        self.x = 10  # position horizontale de départ pour dessiner les formes
        self.y = 10  # position verticale de départ pour dessiner les formes
        self.mH = 0  # hauteur maximale des formes
        self.shapes_top = []  # liste des formes en haut de la ligne
        self.shapes_bottom = []  # liste des formes en bas de la ligne
        self.encoded_matrix = encoded_matrix.copy()  # matrice encodée représentant le message

    def draw_encoded_matrix(self):
        for row in self.encoded_matrix:
            maxWidth = 0
            maxHeight = 0

            for sym_str in row:
                for sym in sym_str:
                    sym_value = ENCODING_VALUES[sym]
                    width = sym_value
                    height = 1

                    if sym_value > 10:
                        width = 10
                        height = sym_value / 10
                    if width > maxWidth:
                        maxWidth = width

                    self.shapes_top.append(Shape(self.x, self.y, self.x + width, self.y + height, 'black'))
                    self.y += height + 2

                    if self.y + 14 > maxHeight:
                        maxHeight = self.y + 14

                self.y += 2
                self.shapes_top.append(Shape(self.x + 3, self.y, self.x + 7, self.y + 4, 'red', True))
                self.y += 8

            if maxHeight > self.mH:
                self.mH = maxHeight

            self.y = 10
            self.x += maxWidth + 5

        self.y = int(self.mH * 2)

        self.x = 10

        self.encoded_matrix.reverse()
        for row in self.encoded_matrix:
            row.reverse()

        for row in self.encoded_matrix:
            maxWidth = 0

            for sym_str in row:
                for sym in sym_str:
                    sym_value = ENCODING_VALUES[sym]

                    width = sym_value
                    height = 1

                    if sym_value > 10:
                        width = 10
                        height = sym_value / 10

                    if width > maxWidth:
                        maxWidth = width
                    self.shapes_bottom.append(Shape(self.x, self.y - height, width, height, 'black'))

                    self.y -= height + 2

                self.y -= 6
                self.shapes_bottom.append(Shape(self.x + 3, self.y, self.x + 7, self.y + 4, 'red', True))
                self.y -= 4

            self.y = int(self.mH * 2)
            self.x += maxWidth + 5

        self.x += 1
        self.y = 2 * self.mH + 6

        img = Image.new('RGB', (int(self.x), int(self.y)), color='white')
        img = ImageOps.expand(img, border=2, fill='black')

        draw = ImageDraw.Draw(img)

        for shape in self.shapes_top:
            if shape.isCircle:
                draw.ellipse((shape.x, shape.y, shape.width - 1, shape.height - 1), fill=shape.color, width=0)
            else:
                draw.rectangle((shape.x, shape.y, shape.width - 1, shape.height - 1), fill=shape.color, width=0)

        for shape in self.shapes_bottom:
            if shape.isCircle:
                draw.ellipse((shape.x, shape.y, shape.width - 1, shape.height - 1), fill=shape.color, width=0)
            else:
                draw.rectangle((shape.x, shape.y, shape.x + shape.width - 1, shape.y + shape.height - 1), fill=shape.color, width=0)

        draw.rectangle((0, 0, 4, 4), fill='black', width=0)  #
        draw.rectangle((0, self.y - 1, 4, self.y + 4), fill='black', width=0)

        img.save('message.png')
        img.show()



msg = input("Please enter a message to encode: ")

encoded_matrix = encode_with_matrix_A(msg)
print("The encoded message is:\n", np.matrix(encoded_matrix), "\n")

decoded_message = decode_with_matrix_A(encoded_matrix)
print("The decoded message is:\n", decoded_message, "\n")

md = MessageDrawer(encoded_matrix)
md.draw_encoded_matrix()
