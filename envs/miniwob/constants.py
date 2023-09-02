TASK_WIDTH = 156
TASK_HEIGHT = 150
TASK_WIDTH_OFFSET = 2
TASK_HEIGHT_OFFSET = 52
NUM_INSTANCES = 16

NUM_DEMOS = 2000

ASCII_CHARSET = "".join(chr(x) for x in range(32, 128))
TEXT_MAX_LENGTH = 256
QUESTIONS = [
  "Is there an email from name?",
  "Is there an email from name with a subject line about 'subject'?",
  "Is there an email from name about 'bodyslice'?",
] + [
  f"Is the {'most' if email_idx == 0 else '2nd' if email_idx == 1 else '3rd' if email_idx == 2 else 'least' if email_idx + 1 == 12 else f'{email_idx+1}th'} recent email from name?" for email_idx in range(12)
] + [
  f"Do I have {n} emails in my inbox?" for n in range(4, 12 + 1)
] + [
  f"Is the {'1st' if email_idx == 0 else '2nd' if email_idx == 1 else '3rd' if email_idx == 2 else f'{email_idx+1}th'} email small?" for email_idx in range(10)
] + [
  f"Is the {'1st' if email_idx == 0 else '2nd' if email_idx == 1 else '3rd' if email_idx == 2 else f'{email_idx+1}th'} email medium?" for email_idx in range(10)
] + [
  f"Is the {'1st' if email_idx == 0 else '2nd' if email_idx == 1 else '3rd' if email_idx == 2 else f'{email_idx+1}th'} email large?" for email_idx in range(10)
]

# PEOPLE_NAMES = ['Adore', 'Adrian', 'Aeriel', 'Ailina', 'Aleda', 'Amargo', 'Angelika', 'Anna', 'Austin', 'Barrie', 'Bird', 'Bride', 'Carline', 'Cathe', 'Celestia', 'Christine', 'Cindee', 'Consuelo', 'Corrina', 'Corry', 'Cthrine', 'Daryn', 'Desdemona', 'Diane-Marie', 'Diann', 'Dina', 'Dolly', 'Doralyn', 'Doralynne', 'Doris', 'Dorothy', 'Edy', 'Ellene', 'Elwira', 'Emmye', 'Florette', 'Frannie', 'Freddy', 'Gail', 'Gianina', 'Giustina', 'Hazel', 'Hedvig', 'Ingaberg', 'Izabel', 'Jaine', 'Jasmin', 'Jerrie', 'Jessica', 'Joby', 'Josie', 'Joy', 'Juieta', 'Kassandra', 'Krystle', 'Kylynn', 'Lolita', 'Lu', 'Lydie', 'Lynda', 'Marline', 'Marlo', 'Marnia', 'Merilyn', 'Mirella', 'Misha', 'Mozelle', 'Myrilla', 'Nana', 'Nell', 'Nicola', 'Noella', 'Nona', 'Pauly', 'Phillis', 'Pietra', 'Polly', 'Quintilla', 'Raynell', 'Reyna', 'Riane', 'Rosene', 'Ruby', 'Sacha', 'Salomi', 'Selia', 'Sephira', 'Shelli', 'Stacia', 'Starlin', 'Suzanne', 'Tabbatha', 'Tally', 'Tania', 'Tiff', 'Tony', 'Twila', 'Umeko', 'Verile', 'Yalonda']
PEOPLE_NAMES = ['Adrian', 'Angelika', 'Bride', 'Celestia', 'Daryn', 'Diane-Marie', 'Doralyn', 'Doralynne', 'Dorothy', 'Florette', 'Giustina', 'Ingaberg', 'Kassandra', 'Lolita', 'Lu', 'Mozelle', 'Myrilla', 'Nicola', 'Reyna', 'Tania']
#LOREM_WORDS = ['a', 'netus', 'ipsum', 'vestibulum', 'justo,', 'elementum,', 'auctor', 'risus,', 'praesent', 'sapien', 'rutrum', 'elementum', 'velit,', 'feugiat', 'duis', 'porta', 'condimentum', 'diam,', 'in', 'nulla', 'cras', 'cum', 'montes,', 'tempor', 'ac', 'morbi', 'posuere', 'mauris', 'commodo', 'libero,', 'consectetur', 'mattis', 'habitasse', 'dolor,', 'aliquam,', 'dolor', 'nunc', 'purus', 'semper', 'augue', 'arcu', 'ultrices', 'lacus', 'sociis', 'tellus', 'ullamcorper', 'parturient', 'sodales', 'tempus', 'venenatis', 'tellus,', 'tristique', 'a,', 'est', 'sed', 'nunc,', 'hendrerit', 'maecenas', 'dui', 'eget', 'etiam', 'ut', 'integer', 'quisque', 'quis', 'aenean', 'venenatis,', 'interdum', 'mi', 'dis', 'enim', 'fringilla', 'amet,', 'sagittis', 'tortor', 'magnis', 'proin', 'gravida', 'accumsan', 'pellentesque', 'tempor,', 'tempus,', 'diam', 'penatibus', 'elit,', 'placerat', 'justo', 'sollicitudin', 'laoreet', 'vehicula', 'vitae', 'dictumst', 'id', 'habitant', 'enim,', 'ante', 'eleifend', 'et', 'massa,', 'volutpat,', 'nisi', 'leo', 'molestie', 'phasellus', 'nisl,', 'felis', 'tortor,', 'platea', 'varius', 'dui,', 'potenti', 'luctus', 'ornare', 'scelerisque', 'nisl', 'pulvinar', 'ridiculus', 'egestas', 'curabitur', 'elit', 'nullam', 'facilisi', 'facilisis', 'orci', 'dignissim', 'libero', 'metus,', 'vel', 'mus', 'est,', 'pretium,', 'lacinia', 'vel,', 'nibh', 'lorem', 'vivamus', 'mauris,', 'eros', 'sapien,', 'sem', 'tincidunt', 'natoque', 'lacus,', 'magna', 'fusce', 'suspendisse', 'arcu,', 'velit', 'non', 'fermentum', 'urna', 'aliquet', 'ultricies', 'leo,', 'eu', 'lobortis', 'rhoncus,', 'risus', 'dictum', 'vitae,', 'iaculis', 'eros,', 'hac', 'consequat,', 'ipsum,', 'faucibus', 'urna,', 'volutpat', 'accumsan,', 'convallis', 'nam', 'vestibulum,', 'non,', 'rhoncus', 'at', 'amet', 'imperdiet', 'consequat', 'congue', 'bibendum', 'turpis', 'pharetra,', 'quam', 'viverra', 'aliquam', 'mollis', 'ligula', 'felis,', 'ac,', 'mi,', 'eu,', 'neque,', 'fermentum,', 'donec', 'neque', 'porttitor', 'pretium', 'nec', 'dapibus', 'adipiscing', 'metus', 'nec,', 'suscipit', 'fames', 'sit', 'malesuada', 'orci,', 'senectus', 'nascetur', 'purus,', 'cursus', 'nisi,', 'blandit', 'sagittis,', 'et,', 'commodo,', 'odio', 'vulputate', 'pharetra', 'massa', 'erat', 'lectus', 'euismod']
LOREM_WORDS = ['aliquam', 'auctor', 'augue', 'condimentum', 'cras', 'cras', 'cursus', 'dis', 'egestas', 'elementum', 'elit', 'enim', 'felis', 'iaculis', 'iaculis', 'iaculis', 'lacus', 'massa', 'natoque', 'nunc', 'parturient', 'penatibus', 'scelerisque', 'suscipit', 'suscipit', 'tempus', 'tortor', 'vel', 'velit', 'venenatis']
HTML_TOKENS = ['ingaberg', 'v', 'id=email-bar>', 'temp', 'par', 'parturi', 'cursus', 'class=email-thread>', 'c', 'partu', '<span', 'ma', 'primary', 'eni', 'class=email-subject>', 'adrian', 'parturien', '<h2>', 'elit', '.', 'giustina', 'au', 'ia', 'scelerisq', 'ven', 'auc', 'doralynne', 'diane-marie', 'nun', 'condimentum', 'p', 'en', 'f', 'scelerisqu', 'lolita', 'class=icon>', '<body>', 'iaculis', 'florette', 'elementum', 'iacu', 'e', 'feli', 'tempus', 'sceleri', 's', 'id=email>', 'tempu', 'di', 'suscipit', 'co', 'i', 'lacu', 'penat', 'celestia', '<div', 'con', 'condimen', 'penati', 'class=email-right>', 'nicola', 'vene', 'scelerisque', 'sc', 'id=area>', 'cra', 'part', 'l', 'angelika', 'class=trash>', 'venenatis', 'id=main>', 'forward', 'eli', 'condimentu', 'pa', 'cu', 'nato', 'condim', 'parturie', 'elemen', 'cond', 'class=email-left>', 'condime', 'dorothy', 'tor', 'id=close-email>', 'aliqu', 'na', 'penatibus', 'fel', 'myrilla', 'augue', 'scel', 't', 'reyna', 'cur', 'vel', 'felis', 've', 'ege', 'enim', 'mas', 'auct', 'la', 'tem', 'sce', 'class=email-forward>', 'egestas', 'elem', 'eg', 'nunc', 'venen', 'lu', 'm', 'class=email-reply>', 'partur', 'me', 'lac', '</h2>', 'class=email-header>', 'natoq', 'su', 'nu', 'ali', 'class=email-body>', 'cr', 'daryn', 'velit', 'penatib', 'cursu', 'curs', 'pe', 'element', 'id=wrap>', 'eleme', 'kassandra', 'reply', 'aucto', 'egesta', 'aliq', 'tortor', 'dis', 'class=email-sender>', 'iaculi', 'cras', 'egest', 'tania', 'iacul', 'scele', 'mass', 'id=open-search>', 'elementu', 'bride', 'doralyn', 'te', 'susc', 'class=email-actions>', 'pen', 'class=star>', 'natoqu', 'condi', 'aug', 'fe', 'condiment', 'aliquam', 'pena', '</div>', 'aliqua', 'to', 'auctor', '</span>', 'el', 'id=main-header>', 'al', 'a', 'penatibu', 'parturient', 'sceleris', 'natoque', '<div>', 'class=email-send>', 'tort', 'sus', 'venenati', 'venena', 'veli', 'venenat', 'torto', 'eges', 'n', 'mozelle', 'nat', 'sceler', 'massa', 'iac', 'ele', 'd', '</body>', 'augu', 'lacus', 'susci', 'suscipi', 'suscip']