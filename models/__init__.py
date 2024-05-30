from pldag import Puan
from dataclasses import dataclass, field
from puan_db_parser import Parser
from typing import Callable
from lexer import ACTION_SET_LIST_COMPOSITE, ACTION_SET_VALUE_COMPOSITE

@dataclass
class PuanDbModel:
    """
        View model for Puan
    """
    model:      Puan

    # Keep track of all queries
    _record:    dict = field(default_factory=dict)
    # Keep track of all set operations
    _sets:      dict = field(default_factory=dict)

    def query(self, query: str, parser: Parser, lexer: Callable[[str], list]):
        """
            Execute a query
        """
        lexed = lexer(query)
        exec_node = parser.parse(lexed)
        result = exec_node()
        self._record[query] = result
        if type(result[-1]) == str and type(lexed[-1]) in [ACTION_SET_LIST_COMPOSITE, ACTION_SET_VALUE_COMPOSITE]:
            self._sets[result[-1]] = lexed[-1].sub_action.name
        return result