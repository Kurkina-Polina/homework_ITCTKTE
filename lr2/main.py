import re
from enum import Enum, auto
from typing import List, Optional, Tuple, Any, Union
import sys
import pymorphy3
from graphviz import Digraph
from google.colab import files

class TokenType(Enum):
    ACTION = auto()       # найти, покажи, выведи
    OBJECT = auto()       # книги, статьи, журналы
    AUTHOR = auto()       # фамилия автора
    TOPIC = auto()        # тема (после "по", "на", "о")
    YEAR_DIGIT = auto()   # цифры года
    YEAR_WORD = auto()    # "год" и его склонения
    PREPOSITION = auto()  # по, на, о (для тем)
    AFTER = auto()        # после, с
    BEFORE = auto()       # до
    IN = auto()           # в, за (для времени)
    AND = auto()          # и
    OR = auto()           # или
    SEP = auto()          # ,
    END = auto()          # конец строки
    UNKNOWN = auto()      # неизвестный токен
    ALL = auto()          # "все"
    PUBLISHED = auto()    # для "изданные"

class Token:
    def __init__(self, type_: TokenType, value: str, position: int):
        self.type = type_
        self.value = value
        self.position = position

    def __repr__(self):
        return f"{self.type.name}('{self.value}')"

class Lexer:
    KEYWORDS = {
        'найти': TokenType.ACTION,
        'показать': TokenType.ACTION,
        'вывести': TokenType.ACTION,
        'отобразить': TokenType.ACTION,
        'искать': TokenType.ACTION,
        'посмотреть': TokenType.ACTION,
        'найди': TokenType.ACTION,

        'книги': TokenType.OBJECT,
        'статьи': TokenType.OBJECT,
        'журналы': TokenType.OBJECT,
        'доклады': TokenType.OBJECT,
        'сборники': TokenType.OBJECT,

        'толстой': TokenType.AUTHOR,
        'достоевский': TokenType.AUTHOR,
        'пушкин': TokenType.AUTHOR,
        'булгаков': TokenType.AUTHOR,
        'чехов': TokenType.AUTHOR,
        'тургенев': TokenType.AUTHOR,
        'гоголь': TokenType.AUTHOR,
        'лермонтов': TokenType.AUTHOR,

        'программирование': TokenType.TOPIC,
        'лингвистика': TokenType.TOPIC,
        'математика': TokenType.TOPIC,
        'медицина': TokenType.TOPIC,
        'алгоритмы': TokenType.TOPIC,
        'физика': TokenType.TOPIC,
        'история': TokenType.TOPIC,
        'биология': TokenType.TOPIC,
        'химия': TokenType.TOPIC,
        'информатика': TokenType.TOPIC,
        'астрономия': TokenType.TOPIC,
        'философия': TokenType.TOPIC,
        'кибернетика': TokenType.TOPIC,
        'агрономия': TokenType.TOPIC,
        'спорт': TokenType.TOPIC,
        'музыка': TokenType.TOPIC,

        'и': TokenType.AND,
        'или': TokenType.OR,

        'по': TokenType.PREPOSITION,
        'на': TokenType.PREPOSITION,
        'о': TokenType.PREPOSITION,

        'после': TokenType.AFTER,
        'с': TokenType.AFTER,
        'до': TokenType.BEFORE,
        'в': TokenType.IN,
        'за': TokenType.IN,

        'год': TokenType.YEAR_WORD,

        'все': TokenType.ALL,
        'изданный': TokenType.PUBLISHED,
    }

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.tokens = []
        self.morph = pymorphy3.MorphAnalyzer()
        self.keywords = {word: type_ for word, type_ in self.KEYWORDS.items()}

    def tokenize(self) -> List[Token]:
        while self.pos < len(self.text):
            if self.text[self.pos].isspace() or self.text[self.pos] in ".()":
                self.pos += 1
                continue

            # <year_digit> ::= ["-"] <digit> { <digit> }
            match = re.match(r'-?\d+', self.text[self.pos:])
            if match:
                value = match.group()
                self.tokens.append(Token(TokenType.YEAR_DIGIT, value, self.pos))
                self.pos += len(value)
                continue

            match = re.match(r'[а-яёА-ЯЁ\-]+', self.text[self.pos:])
            if match:
                value = match.group()
                parsed = self.morph.parse(value)[0]
                normal_form = parsed.normal_form.lower()

                token_type = None

                if value.lower() in self.keywords:
                    token_type = self.keywords[value.lower()]
                elif normal_form in self.keywords:
                    token_type = self.keywords[normal_form]

                if token_type is None:
                    token_type = TokenType.UNKNOWN

                self.tokens.append(Token(token_type, value, self.pos))
                self.pos += len(value)
                continue

            if self.text[self.pos] == ',':
                self.tokens.append(Token(TokenType.SEP, ',', self.pos))
                self.pos += 1
                continue

            raise SyntaxError(f"Неизвестный символ '{self.text[self.pos]}' в позиции {self.pos}")

        self.tokens.append(Token(TokenType.END, '', self.pos))
        return self.tokens

class Node:
    def __repr__(self, level=0):
        return "  " * level + self.__class__.__name__

class QueryNode(Node):
    def __init__(self, action: 'ActionNode', object_spec: 'ObjectSpecNode',
                 additional_groups: List[Tuple[str, 'ObjectSpecNode']] = None):
        self.action = action
        self.object_spec = object_spec
        self.additional_groups = additional_groups or []

    def __repr__(self, level=0):
        lines = [super().__repr__(level)]
        lines.append(self.action.__repr__(level + 1))
        lines.append(self.object_spec.__repr__(level + 1))
        for conj, spec in self.additional_groups:
            lines.append("  " * (level + 1) + f"Conj: {conj}")
            lines.append(spec.__repr__(level + 1))
        return '\n'.join(lines)

class ActionNode(Node):
    def __init__(self, value: str):
        self.value = value

    def __repr__(self, level=0):
        return "  " * level + f"Action: {self.value}"

class ObjectSpecNode(Node):
    def __init__(self, object_type: 'ObjectTypeNode', filters: List[Node] = None):
        self.object_type = object_type
        self.filters = filters or []

    def __repr__(self, level=0):
        lines = [super().__repr__(level)]
        lines.append(self.object_type.__repr__(level + 1))
        if self.filters:
            lines.append("  " * (level + 1) + "Filters:")
            for f in self.filters:
                lines.append(f.__repr__(level + 2))
        return '\n'.join(lines)

class ObjectTypeNode(Node):
    def __init__(self, value: str):
        self.value = value

    def __repr__(self, level=0):
        return "  " * level + f"ObjectType: {self.value}"

class AuthorFilterNode(Node):
    def __init__(self, value: str):
        self.value = value

    def __repr__(self, level=0):
        return "  " * level + f"AuthorFilter: {self.value}"

class TopicFilterNode(Node):
    def __init__(self, preposition: str, topic: str):
        self.preposition = preposition
        self.topic = topic

    def __repr__(self, level=0):
        return "  " * level + f"TopicFilter: {self.preposition} {self.topic}"
class YearFilterNode(Node):
    def __init__(self, relation: str, year: str):
        self.relation = relation
        self.year = year

    def __repr__(self, level=0):
        return "  " * level + f"YearFilter: {self.relation} {self.year}"

class ParserError(Exception):
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"{message} на токене '{token.value}' (позиция {token.position})")

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # END

    def consume(self, expected_type: Optional[TokenType] = None, expected_value: Optional[str] = None) -> Token:
        token = self.current_token()
        if expected_type is not None and token.type != expected_type:
            raise ParserError(f"Ожидался {expected_type.name}, получен {token.type.name}", token)
        if expected_value is not None and token.value.lower() != expected_value.lower():
            raise ParserError(f"Ожидалось значение '{expected_value}', получено '{token.value}'", token)
        self.pos += 1
        return token

    def peek(self) -> Token:
        return self.current_token()

    def parse(self) -> QueryNode:
        # <qwery> ::= <command> <object_spec> { <sep><object_group> }*
        action = self.parse_action()
        object_spec = self.parse_object_spec()

        # { <sep><object_group> }*
        additional_groups = []
        while self.peek().type in (TokenType.SEP):
            self.consume()
            next_spec = self.parse_object_spec()
            additional_groups.append(next_spec)

        if self.peek().type != TokenType.END:
            raise ParserError("Ожидался конец строки, но есть лишние токены", self.peek())

        return QueryNode(action, object_spec, additional_groups)

    def parse_action(self) -> ActionNode:
        token = self.consume(expected_type=TokenType.ACTION)
        return ActionNode(token.value)

    def parse_object_spec(self) -> ObjectSpecNode:
        # Пропускаем "все" если есть
        if self.peek().type == TokenType.ALL:
            self.consume(TokenType.ALL)

        # <object_spec> ::= <object> [ <filters> ]
        obj_token = self.consume(expected_type=TokenType.OBJECT)
        object_type = ObjectTypeNode(obj_token.value)

        # [ <filters> ]
        # <filters> ::= <filter> { <conj><filter> }*
        filters = []

        if self._is_filter_start():
            filters.append(self.parse_filter())

            while self.peek().type in (TokenType.AND, TokenType.OR):
                conj_token = self.consume()
                filters.append(self.parse_filter())

        return ObjectSpecNode(object_type, filters)

    def _is_filter_start(self) -> bool:
        t = self.peek().type
        return t in (TokenType.AUTHOR, TokenType.PREPOSITION,
                     TokenType.AFTER, TokenType.BEFORE, TokenType.IN)

    def parse_filter(self) -> Node:
        # <filter> ::= <author> | <topic_filter> | <year_filter>
        token = self.peek()

        if token.type == TokenType.PUBLISHED:
            self.consume(TokenType.PUBLISHED)  # пропускаем "изданные"
            return self.parse_filter()
        elif token.type == TokenType.YEAR_DIGIT:
            # Прямой год без предлога
            year_digit = self.consume(TokenType.YEAR_DIGIT)
            if self.peek().type == TokenType.YEAR_WORD:
                self.consume(TokenType.YEAR_WORD)
            return YearFilterNode("IN", year_digit.value)
        elif token.type == TokenType.AUTHOR:
            return self.parse_author_filter()
        elif token.type == TokenType.PREPOSITION:
            return self.parse_topic_filter()
        elif token.type in (TokenType.AFTER, TokenType.BEFORE, TokenType.IN):
            return self.parse_year_filter()
        else:
            raise ParserError("Неизвестный тип фильтра", token)

    def parse_author_filter(self) -> AuthorFilterNode:
        # <author> ::= ...
        token = self.consume(TokenType.AUTHOR)
        return AuthorFilterNode(token.value)

    def parse_topic_filter(self) -> TopicFilterNode:
        # <topic_filter> ::= <preposition> <topic>
        prep_token = self.consume(TokenType.PREPOSITION)
        topic_token = self.consume(TokenType.TOPIC)
        return TopicFilterNode(prep_token.value, topic_token.value)

    def parse_year_filter(self) -> YearFilterNode:
        # <year_filter> ::= <after> | <before> | <in> <year_digit> <year_word>
        token = self.peek()
        relation = token.value

        if token.type == TokenType.AFTER:
            self.consume(TokenType.AFTER)
        elif token.type == TokenType.BEFORE:
            self.consume(TokenType.BEFORE)
        elif token.type == TokenType.IN:
            self.consume(TokenType.IN)

        year_digit = self.consume(TokenType.YEAR_DIGIT)

        if self.peek().type == TokenType.YEAR_WORD:
            self.consume(TokenType.YEAR_WORD)

        return YearFilterNode(relation, year_digit.value)

def parse_query(query: str) -> Tuple[bool, Optional[QueryNode], Optional[str]]:
    try:
        lexer = Lexer(query.strip())
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        tree = parser.parse()

        return True, tree, None
    except (SyntaxError, ParserError) as e:
        return False, None, str(e)

def main():
    print("Загрузите текстовый файл с запросами (по одному на строку).")
    from google.colab import files
    uploaded = files.upload()

    for filename in uploaded.keys():
        content = uploaded[filename].decode('utf-8')
        lines = content.strip().split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            print("-" * 40)
            print(f"Запрос #{i}: {line}")
            success, tree, error = parse_query(line)
            if success:
                print("Статус: успех")
                print("Дерево разбора:")
                print(tree)
                visualizer = ASTVisualizer()
                visualizer.visualize(tree)
                visualizer.save(f'query_{i}')
            else:
                print("Статус: неудача")
                print(f"Ошибка: {error}")

# Для визуализации
from IPython.display import Image, display
class ASTVisualizer:
    def __init__(self):
        self.dot = Digraph(comment='Query AST')
        self.dot.attr(rankdir='TB')  # Сверху вниз

    def visualize(self, node, parent_id=None):
        node_id = str(id(node))
        node_label = self._get_label(node)
        self.dot.node(node_id, node_label)

        if parent_id:
            self.dot.edge(parent_id, node_id)

        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue
            try:
                attr_value = getattr(node, attr_name)
            except:
                continue

            if isinstance(attr_value, Node):
                self.visualize(attr_value, node_id)
            elif isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, Node):
                        self.visualize(item, node_id)
                    elif isinstance(item, tuple):
                        # Для кортежей (conj, spec)
                        for sub_item in item:
                            if isinstance(sub_item, Node):
                                self.visualize(sub_item, node_id)

        return self.dot

    def _get_label(self, node):
        if isinstance(node, QueryNode):
            return f"Query\n{node.action.value}"
        elif isinstance(node, ActionNode):
            return f"Action\n{node.value}"
        elif isinstance(node, ObjectSpecNode):
            return f"ObjectSpec\n{node.object_type.value}"
        elif isinstance(node, ObjectTypeNode):
            return f"Type\n{node.value}"
        elif isinstance(node, AuthorFilterNode):
            return f"Author\n{node.value}"
        elif isinstance(node, TopicFilterNode):
            return f"Topic\n{node.preposition} {node.topic}"
        elif isinstance(node, YearFilterNode):
            return f"Year\n{node.relation} {node.year}"
        return node.__class__.__name__

    def save(self, filename='ast_output'):
        self.dot.render(filename, format='png', cleanup=True, view=False)
        display(Image(f'{filename}.png'))


if __name__ == "__main__":
    main()