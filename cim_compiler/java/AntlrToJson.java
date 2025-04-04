import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.ArrayList;
// import gson
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

public class AntlrToJson {

  private static final Gson PRETTY_PRINT_GSON = new GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create();
  private static final Gson GSON = new Gson();

  public static String toJson(ParseTree tree) {
    return toJson(tree, true);
  }

  public static String toJson(ParseTree tree, boolean prettyPrint) {
    return prettyPrint ? PRETTY_PRINT_GSON.toJson(toMap(tree)) : GSON.toJson(toMap(tree));
  }

  public static Map<String, Object> toMap(ParseTree tree) {
    Map<String, Object> map = new LinkedHashMap<>();
    traverse(tree, map);
    return map;
  }

  public static void traverse(ParseTree tree, Map<String, Object> map) {

    if (tree instanceof TerminalNodeImpl) {
      Token token = ((TerminalNodeImpl) tree).getSymbol();
      map.put("type", token.getType());
      map.put("text", token.getText());
    }
    else {
      List<Map<String, Object>> children = new ArrayList<>();
      String name = tree.getClass().getSimpleName().replaceAll("Context$", "");
      map.put(Character.toLowerCase(name.charAt(0)) + name.substring(1), children);

      for (int i = 0; i < tree.getChildCount(); i++) {
        Map<String, Object> nested = new LinkedHashMap<>();
        children.add(nested);
        traverse(tree.getChild(i), nested);
      }
    }
  }

  public static void main(String[] args) {
    // system.out.println(args[0]);
    // System.out.println("args.length: " + args.length);
    // System.out.println("read cim code from " + args[0]);
    // System.out.println("write json to " + args[1]);
    String srcFilePath = args[0];
    String dstFilePath = args[1];
    try {
        String source = new String(Files.readAllBytes(Paths.get(srcFilePath)), StandardCharsets.UTF_8);
        CIMLexer lexer = new CIMLexer(CharStreams.fromString(source));
        CIMParser parser = new CIMParser(new CommonTokenStream(lexer));
        String json = toJson(parser.program());
        // System.out.println(json);
        // write to file
        Files.write(Paths.get(dstFilePath), json.getBytes());
    } catch (IOException e) {
        e.printStackTrace();
    }
  }
}