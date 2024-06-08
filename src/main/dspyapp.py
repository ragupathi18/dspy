from dspy.datasets import HotPotQA
from dspy import Example
from customllm import Claude
import dspy

#dspy.settings.configure()

turbo=Claude("claude","dummy")



colbert=dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=turbo, rm=colbert)

retrieve=dspy.Retrieve(k=3)

topK_passages=retrieve("What is the nationality of Robert Irvine?").passages
print(f"Top {retrieve.k} passages for question: What is the nationality of Robert Irvine? \n",'-'*30,"\n")
for idx, passage in enumerate(topK_passages):
    print(f"{idx+1}]", passage,"\n")





from dspy.datasets import HotPotQA

# Load the dataset.
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

print(len(trainset), len(devset))

train_example = trainset[0]
#print(f"Question: {train_example.question}")
#print(f"Answer: {train_example.answer}")

dev_example = devset[18]
#print(f"Question: {dev_example.question}")
#print(f"Answer: {dev_example.answer}")
#print(f"Relevant Wikipedia Titles: {dev_example.gold_titles}")
#t= [Example(question="At My Window was released by which American singer-songwriter?", answer="John Townes Van Zandt").with_inputs("question")]

#print(t)

#print(f"For this dataset, training examples have input keys {train_example.inputs().keys()} and label keys {train_example.labels().keys()}")
#print(f"For this dataset, dev examples have input keys {dev_example.inputs().keys()} and label keys {dev_example.labels().keys()}")



class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question=dspy.InputField()
    answer=dspy.OutputField(desc="often between 1 and 5 words")

"""
generate_answer=dspy.Predict(BasicQA)
pred=generate_answer(question=dev_example.question)

print(f"Question: {dev_example.question}")
print(f"Predicted Answer: {pred.answer}")

turbo.inspect_history(n=1)
"""
generate_answer_with_chain_of_thought=dspy.ChainOfThought(BasicQA)
pred=generate_answer_with_chain_of_thought(question=dev_example.question)

print(f"Question: {dev_example.question}")
print(f"Thought : {pred.rationale.split('.',1)[1].strip()}")
print(f"Predicted Answer: {pred.answer}")


retrieve=dspy.Retrieve(k=3)

topK_passages=retrieve(dev_example.question).passages
print(f"Top {retrieve.k} passages for question: {dev_example.question} \n",'-'*30,"\n")
for idx, passage in enumerate(topK_passages):
    print(f"{idx+1}]", passage,"\n")


"""
trainset=[Example({'question': 'At My Window was released by which American singer-songwriter?', 'answer': 'John Townes Van Zandt'}) (input_keys={'question'}), Example({'question': 'which  American actor was Candace Kita  guest starred with ', 'answer': 'Bill Murray'}) (input_keys={'question'}), Example({'question': 'Which of these publications was most recently published, Who Put the Bomp or Self?', 'answer': 'Self'}) (input_keys={'question'}), Example({'question': 'The Victorians - Their Story In Pictures is a documentary series written by an author born in what year?', 'answer': '1950'}) (input_keys={'question'}), Example({'question': 'Which magazine has published articles by Scott Shaw, Tae Kwon Do Times or Southwest Art?', 'answer': 'Tae Kwon Do Times'}) (input_keys={'question'}), Example({'question': 'In what year was the club founded that played Manchester City in the 1972 FA Charity Shield', 'answer': '1874'}) (input_keys={'question'}), Example({'question': 'Which is taller, the Empire State Building or the Bank of America Tower?', 'answer': 'The Empire State Building'}) (input_keys={'question'}), Example({'question': 'Which American actress who made their film debut in the 1995 teen drama "Kids" was the co-founder of Voto Latino?', 'answer': 'Rosario Dawson'}) (input_keys={'question'}), Example({'question': 'Tombstone stared an actor born May 17, 1955 known as who?', 'answer': 'Bill Paxton'}) (input_keys={'question'}), Example({'question': 'What is the code name for the German offensive that started this Second World War engagement on the Eastern Front (a few hundred kilometers from Moscow) between Soviet and German forces, which included 102nd Infantry Division?', 'answer': 'Operation Citadel'}) (input_keys={'question'}), Example({'question': 'Who acted in the shot film The Shore and is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of "Hamlet." ?', 'answer': 'Kerry Condon'}) (input_keys={'question'}), Example({'question': 'Which company distributed this 1977 American animated film produced by Walt Disney Productions for which Sherman Brothers wrote songs?', 'answer': 'Buena Vista Distribution'}) (input_keys={'question'}), Example({'question': 'Samantha Cristoforetti and Mark Shuttleworth are both best known for being first in their field to go where? ', 'answer': 'space'}) (input_keys={'question'}), Example({'question': 'Having the combination of excellent foot speed and bat speed helped Eric Davis, create what kind of outfield for the Los Angeles Dodgers? ', 'answer': '"Outfield of Dreams"'}) (input_keys={'question'}), Example({'question': 'Which Pakistani cricket umpire who won 3 consecutive ICC umpire of the year awards in 2009, 2010, and 2011 will be in the ICC World Twenty20?', 'answer': 'Aleem Sarwar Dar'}) (input_keys={'question'}), Example({'question': 'The Organisation that allows a community to influence their operation or use and to enjoy the benefits arisingwas founded in what year?', 'answer': '2010'}) (input_keys={'question'}), Example({'question': '"Everything Has Changed" is a song from an album released under which record label ?', 'answer': 'Big Machine Records'}) (input_keys={'question'}), Example({'question': 'Who is older, Aleksandr Danilovich Aleksandrov or Anatoly Fomenko?', 'answer': 'Aleksandr Danilovich Aleksandrov'}) (input_keys={'question'}), Example({'question': 'On the coast of what ocean is the birthplace of Diogal Sakho?', 'answer': 'Atlantic'}) (input_keys={'question'}), Example({'question': 'This American guitarist best known for her work with the Iron Maidens is an ancestor of a composer who was known as what?', 'answer': 'The Waltz King'}) (input_keys={'question'})]

devseet=[Example({'question': 'Are both Cangzhou and Qionghai in the Hebei province of China?', 'answer': 'no', 'gold_titles': {'Cangzhou', 'Qionghai'}}) (input_keys={'question'}), Example({'question': 'Who conducts the draft in which Marc-Andre Fleury was drafted to the Vegas Golden Knights for the 2017-18 season?', 'answer': 'National Hockey League', 'gold_titles': {'2017–18 Pittsburgh Penguins season', '2017 NHL Expansion Draft'}}) (input_keys={'question'}), Example({'question': 'The Wings entered a new era, following the retirement of which Canadian retired professional ice hockey player and current general manager of the Tampa Bay Lightning of the National Hockey League (NHL)?', 'answer': 'Steve Yzerman', 'gold_titles': {'Steve Yzerman', '2006–07 Detroit Red Wings season'}}) (input_keys={'question'}), Example({'question': 'What river is near the Crichton Collegiate Church?', 'answer': 'the River Tyne', 'gold_titles': {'Crichton Collegiate Church', 'Crichton Castle'}}) (input_keys={'question'}), Example({'question': 'In the 10th Century A.D. Ealhswith had a son called Æthelweard by which English king?', 'answer': 'King Alfred the Great', 'gold_titles': {'Æthelweard (son of Alfred)', 'Ealhswith'}}) (input_keys={'question'}), Example({'question': 'The Newark Airport Exchange is at the northern edge of an airport that is operated by whom?', 'answer': 'Port Authority of New York and New Jersey', 'gold_titles': {'Newark Airport Interchange', 'Newark Liberty International Airport'}}) (input_keys={'question'}), Example({'question': 'Where did an event take place resulting in a win during a domestic double due to the action of a Peruvian footballer known for his goal scoring ability?', 'answer': 'Bundesliga', 'gold_titles': {'2005–06 FC Bayern Munich season', 'Claudio Pizarro'}}) (input_keys={'question'}), Example({'question': 'Are both Chico Municipal Airport and William R. Fairchild International Airport in California?', 'answer': 'no', 'gold_titles': {'Chico Municipal Airport', 'William R. Fairchild International Airport'}}) (input_keys={'question'}), Example({'question': 'In which Maine county is Fort Pownall located?', 'answer': 'Waldo County, Maine', 'gold_titles': {'Fort Pownall', 'Stockton Springs, Maine'}}) (input_keys={'question'}), Example({'question': 'Which 90s rock band has more recently reformed, Gene or The Afghan Whigs?', 'answer': 'The Afghan Whigs', 'gold_titles': {'Gene (band)', 'The Afghan Whigs'}}) (input_keys={'question'}), Example({'question': 'What year did the mountain known in Italian as "Monte Vesuvio", erupt?', 'answer': '79 AD', 'gold_titles': {'Curse of the Faceless Man', 'Mount Vesuvius'}}) (input_keys={'question'}), Example({'question': 'Is the 72nd field brigade part of the oldest or newest established field army?', 'answer': 'the oldest', 'gold_titles': {'First United States Army', '72nd Field Artillery Brigade (United States)'}}) (input_keys={'question'}), Example({'question': 'Was Stanislaw Kiszka paid for his services by the Royal Treasury?', 'answer': 'not', 'gold_titles': {'Hetmans of the Polish–Lithuanian Commonwealth', 'Stanisław Kiszka'}}) (input_keys={'question'}), Example({'question': 'Which film director is younger, Del Lord or Wang Xiaoshuai?', 'answer': 'Del Lord', 'gold_titles': {'Del Lord', 'Wang Xiaoshuai'}}) (input_keys={'question'}), Example({'question': 'Lord North Street has a resident in which former Conservative MP who received an 18-month prison sentence for perjury in 1999?', 'answer': 'Jonathan William Patrick Aitken', 'gold_titles': {'Lord North Street', 'Jonathan Aitken'}}) (input_keys={'question'}), Example({'question': 'What is the name of this region of Italy, referring to the medieval March of Ancona and nearby marches of Camerino and Fermo, where the comune Pollenza is located?', 'answer': 'Marche', 'gold_titles': {'Pollenza', 'Marche'}}) (input_keys={'question'}), Example({'question': 'William Hughes Miller was born in a city with how many inhabitants ?', 'answer': '7,402 at the 2010 census', 'gold_titles': {'Kosciusko, Mississippi', 'William Hughes Miller'}}) (input_keys={'question'}), Example({'question': 'What do students do at the school of New York University where Meleko Mokgosi is an artist and assistant professor?', 'answer': 'design their own interdisciplinary program', 'gold_titles': {'Meleko Mokgosi', 'Gallatin School of Individualized Study'}}) (input_keys={'question'}), Example({'question': 'What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?', 'answer': 'English', 'gold_titles': {'Restaurant: Impossible', 'Robert Irvine'}}) (input_keys={'question'}), Example({'question': 'What American actor plays an East side drug lord who prefers peaceful solutions to business disputes when possible?', 'answer': 'Robert F. Chew', 'gold_titles': {'Robert F. Chew', 'Proposition Joe'}}) (input_keys={'question'}), Example({'question': 'What city is 11 miles north of the birthplace of actor Toby Sawyer?', 'answer': 'Manchester', 'gold_titles': {'Wilmslow', 'Toby Sawyer'}}) (input_keys={'question'}), Example({'question': 'Who was born first, Tony Kaye or Deepa Mehta?', 'answer': 'Deepa Mehta', 'gold_titles': {'Deepa Mehta', 'Tony Kaye (director)'}}) (input_keys={'question'}), Example({'question': "What is the English translation of the name of the store that Macy's replaced in Boise Town Square?", 'answer': 'the good market', 'gold_titles': {'Boise Towne Square', 'The Bon Marché'}}) (input_keys={'question'}), Example({'question': 'Who did Lizzette Reynolds fire to make her notable in November 2007?', 'answer': 'Christine Comer', 'gold_titles': {'Lizzette Reynolds', 'Christine Comer'}}) (input_keys={'question'}), Example({'question': 'What was the name of the man, who was billed by the coiner of the phrase "There\'s a sucker born every minute" as "Boy Lightning Calculator"?', 'answer': 'William Street Hutchings', 'gold_titles': {'William S. Hutchings', 'P. T. Barnum'}}) (input_keys={'question'}), Example({'question': "Which battle was fought for a shorter period of time, the Battle of the Ch'ongch'on River, or the Meuse-Argonne Offensive?", 'answer': "Battle of the Ch'ongch'on River", 'gold_titles': {'Meuse-Argonne Offensive', "Battle of the Ch'ongch'on River"}}) (input_keys={'question'}), Example({'question': 'What cricketeer active 1974–1993 had a strong performance in the tour by the Australian cricket team in England in 1981?', 'answer': 'Ian Botham', 'gold_titles': {'Australian cricket team in England in 1981', 'Ian Botham'}}) (input_keys={'question'}), Example({'question': 'What is the present post of the head coach of the 1982 NC State Wolfpack football team ?', 'answer': 'defensive assistant at Florida Atlantic', 'gold_titles': {'1982 NC State Wolfpack football team', 'Monte Kiffin'}}) (input_keys={'question'}), Example({'question': 'Which Scottish actor sang "Come What May"?', 'answer': 'Ewan McGregor', 'gold_titles': {'Ewan McGregor', 'Come What May (2001 song)'}}) (input_keys={'question'}), Example({'question': 'Where have Ivan Bella and Frank De Winne both traveled?', 'answer': 'space', 'gold_titles': {'Ivan Bella', 'Frank De Winne'}}) (input_keys={'question'}), Example({'question': 'The original work by Anton Chekhov involving a disillusioned schoolmaster, which inspired a later play by this British playwright, was written specifically for whom?', 'answer': 'Maria Yermolova', 'gold_titles': {'Platonov (play)', 'Wild Honey (play)'}}) (input_keys={'question'}), Example({'question': 'Are Roswell International Air Center and Pago Pago International Airport both located in the mainland US?', 'answer': 'no', 'gold_titles': {'Pago Pago International Airport', 'Roswell International Air Center'}}) (input_keys={'question'}), Example({'question': 'Untold: The Greatest Sports Stories Never Told was hosted by a sportscaster commonly referred to as what ?', 'answer': 'the voice of basketball', 'gold_titles': {'Untold: The Greatest Sports Stories Never Told', 'Marv Albert'}}) (input_keys={'question'}), Example({'question': 'Are Walt Disney and Sacro GRA both documentry films?', 'answer': 'yes', 'gold_titles': {'Walt Disney (film)', 'Sacro GRA'}}) (input_keys={'question'}), Example({'question': 'What is the Palestinian Islamic organization that governs th small territory on the eastern coast of the Mediterranean Sea that was captured by Israel during the 1967 Six-Day War?', 'answer': 'Hamas', 'gold_titles': {'Status of territories occupied by Israel in 1967', 'Gaza Strip'}}) (input_keys={'question'}), Example({'question': 'What album did the song of which Taylor Swift premiered the music video of during the pre-show of the 2015 MTV Video Music Awards come from?', 'answer': '1989', 'gold_titles': {'2015 MTV Video Music Awards', 'Wildest Dreams (Taylor Swift song)'}}) (input_keys={'question'}), Example({'question': 'Which is considered a genus level classification, Apera or Gunnera manicata?', 'answer': 'Apera', 'gold_titles': {'Apera', 'Gunnera manicata'}}) (input_keys={'question'}), Example({'question': 'Do The Drums and Pussy Galore play music of similar genres?', 'answer': 'no', 'gold_titles': {'The Drums', 'Pussy Galore (band)'}}) (input_keys={'question'}), Example({'question': 'What is the post-nominal abbreviation for the university where the Banded Mongoose Research Project is based?', 'answer': 'Exon', 'gold_titles': {'Banded Brothers', 'University of Exeter'}}) (input_keys={'question'}), Example({'question': 'Are both Benjamin Christensen and Len Wiseman directors?', 'answer': 'yes', 'gold_titles': {'Len Wiseman', 'Benjamin Christensen'}}) (input_keys={'question'}), Example({'question': 'Steven Cuitlahuac Melendez and Disney are connected by what American animator?', 'answer': 'Bill Melendez', 'gold_titles': {'Steven C. Melendez', 'Bill Melendez'}}) (input_keys={'question'}), Example({'question': 'Shark Creek is located on this river which is in the northern rivers district?', 'answer': 'Clarence River', 'gold_titles': {'Shark Creek, New South Wales', 'Clarence River (New South Wales)'}}) (input_keys={'question'}), Example({'question': 'Who was the producer of the 2016 animated film about an amnesiac fish?', 'answer': 'Pixar', 'gold_titles': {'Hayden Rolence', 'Finding Dory'}}) (input_keys={'question'}), Example({'question': 'Who purchased the team Michael Schumacher raced for in the 1995 Monaco Grand Prix in 2000?', 'answer': 'Renault', 'gold_titles': {'Benetton Formula', '1995 Monaco Grand Prix'}}) (input_keys={'question'}), Example({'question': 'Fredrick Law Olmsted was an American landscape architect, journalist, social critic and public administrator that designed what neighborhood in Trenton, New Jersey?', 'answer': 'Cadwalader Heights', 'gold_titles': {'Frederick Law Olmsted', 'Cadwalader Heights, Trenton, New Jersey'}}) (input_keys={'question'}), Example({'question': 'Gordon Warnecke worked alongside the former senator for which political party on Young Toscanini?', 'answer': '"Forza Italia" party.', 'gold_titles': {'Franco Zeffirelli', 'Gordon Warnecke'}}) (input_keys={'question'}), Example({'question': 'André Zucca was a French photographer who worked with a German propaganda magazine published by what Nazi organization?', 'answer': 'the Wehrmacht', 'gold_titles': {'André Zucca', 'Signal (magazine)'}}) (input_keys={'question'}), Example({'question': 'Both Bill Ponsford and Bill Woodfull played what?', 'answer': 'cricketer', 'gold_titles': {'Bill Woodfull', 'Bill Ponsford'}}) (input_keys={'question'}), Example({'question': ' Suzana S. Drobnjaković Ponti acted in a film loosely based on a book by who?', 'answer': 'Danny Wallace', 'gold_titles': {'Sasha Alexander', 'Yes Man (film)'}}) (input_keys={'question'}), Example({'question': 'In what city was the Election Law Journal founded?', 'answer': 'Portland', 'gold_titles': {'Election Law Journal', 'Reed College'}}) (input_keys={'question'})]



print(trainset)


retrieve = dspy.Retrieve(k=3) """