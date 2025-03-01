import React from "react";
import { Text } from "@chakra-ui/react";
import FancyText from "@carefully-coded/react-text-gradient";

interface WordWidgetProps {
  data: {
    type: 'simpleQuestion' | 'inverseQuestion' | 'sentenceSimpleQuestion' | 'inverseSentenceSimpleQuestion';
    question: {
      value: string
    };
    answer: {
      value: string,
      description: string
    }
  };
  isHiddenWord: boolean;
  setIsHiddenWord: (hidden: boolean) => void;
}

const WordWidget: React.FC<WordWidgetProps> = ({
  data,
  isHiddenWord,
  setIsHiddenWord,
}: WordWidgetProps) => {
  switch (data.type) {
    case 'simpleQuestion':
      return (
        <>
          <Text fontSize="xl">{data.question.value}</Text>
          <FancyText
            gradient={{ from: "#17acff", to: "#ff68f0", type: "linear" }}
            animate
            animateDuration={1000}
          >
            <Text
              style={{
                backgroundColor: isHiddenWord ? "black" : "unset",
                cursor: isHiddenWord ? "pointer" : "unset",
              }}
              onClick={() => setIsHiddenWord(false)}
            >
              {data.answer.value}
            </Text>
          </FancyText>
          <Text style={{
                visibility: isHiddenWord ? "hidden" : "visible",
              }}
          >{data.answer?.description}</Text>
        </>
      );

    case 'inverseQuestion':
      return (
        <>
          <Text fontSize="xl">{data.question.value}</Text>
          <FancyText
            gradient={{ from: "#ff68f0", to: "#17acff", type: "linear" }}
            animate
            animateDuration={1000}
          >
            <Text
              style={{
                backgroundColor: isHiddenWord ? "black" : "unset",
                cursor: isHiddenWord ? "pointer" : "unset",
              }}
              onClick={() => setIsHiddenWord(false)}
            >
              {data.answer.value}
            </Text>
          </FancyText>
          <Text style={{
                visibility: isHiddenWord ? "hidden" : "visible",
              }}
          >{data.answer?.description}</Text>
        </>
      );

    case 'sentenceSimpleQuestion':
      return (
        <>
          <Text>{data.question.value}</Text>
          <FancyText
            gradient={{ from: "#17acff", to: "#ff68f0", type: "linear" }}
            animate
            animateDuration={1000}
          >
            <Text
              style={{
                backgroundColor: isHiddenWord ? "black" : "unset",
                cursor: isHiddenWord ? "pointer" : "unset",
              }}
              onClick={() => setIsHiddenWord(false)}
            >
              {data.answer.value}
            </Text>
          </FancyText>
        </>
      );

      case 'inverseSentenceSimpleQuestion':
        return (
          <>
            <Text>{data.question.value}</Text>
            <FancyText
              gradient={{ from: "#17acff", to: "#ff68f0", type: "linear" }}
              animate
              animateDuration={1000}
            >
              <Text
                style={{
                  backgroundColor: isHiddenWord ? "black" : "unset",
                  cursor: isHiddenWord ? "pointer" : "unset",
                }}
                onClick={() => setIsHiddenWord(false)}
              >
                {data.answer.value}
              </Text>
            </FancyText>
          </>
        );

    default:
      return <Text>Invalid widget type</Text>;
  }
};

export default WordWidget;
